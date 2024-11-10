# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import torch
import tqdm

from src.dust3r.utils.device import collate_with_cat, to_cpu
from src.dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from src.dust3r.utils.misc import invalid_to_nans


def loss_of_one_batch(
    batch, model, criterion, device, precision, symmetrize_batch=False, use_amp=False, ret=None, profiling=False,
):
    """
    Args:
        batch (list[dict]): a list of views, each view is a dict of tensors, the tensors are batched
    """       
    for view in batch:
        for (
            name
        ) in (
            "img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres".split()
        ):  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    views = batch

    autocast_dict = dict(device_type=device.type)
    if precision == "32":
        autocast_dict["enabled"] = False
    elif precision == "16-mixed":
        autocast_dict["dtype"] = torch.float16
    elif precision in ["bf16-mixed", "bf16-mixed-no-grad-scaling"]:
        autocast_dict["dtype"] = torch.bfloat16

    with torch.autocast(**autocast_dict):
        preds = model(views, profiling=profiling)

        # loss is supposed to be symmetric
        loss = (
            criterion(views, preds) if criterion is not None else None
        )

    result = dict(views=views, preds=preds, loss=loss)
    return result[ret] if ret else result


@torch.no_grad()
def inference(multiple_views_in_one_sample, model, device, dtype, verbose=True, profiling=False):
    if verbose:
        print(f">> Inference with model on {len(multiple_views_in_one_sample)} images")
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(multiple_views_in_one_sample))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    res = loss_of_one_batch(
        collate_with_cat([tuple(multiple_views_in_one_sample)]), model, None, device, dtype, profiling=profiling
    )
    result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(imgs):
    shapes = [img["img"].shape[-2:] for img in imgs]
    return all(shape == shapes[0] for shape in shapes)


def get_pred_pts3d(gt, pred, use_pose=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:
        # pts3d from my camera
        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred["pts3d_in_other_view"]  # return!

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
