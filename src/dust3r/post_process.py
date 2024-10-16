# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities for interpreting the DUST3R output
# --------------------------------------------------------
import numpy as np
import torch

from src.dust3r.utils.geometry import xy_grid


def estimate_focal_knowing_depth(
    pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(
        -1, 1, 2
    )  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == "median":
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (
        2 * np.tan(np.deg2rad(60) / 2)
    )  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    # print(focal)
    return focal

def estimate_focal_knowing_depth_and_confidence_mask(
    pts3d, pp, conf_mask, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    This function considers only points where conf_mask is True.
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, H, W, 2) - pp.view(
        -1, 1, 1, 2
    )  # B,H,W,2

    # Apply the confidence mask
    conf_mask = conf_mask.view(B, H, W)  # Ensure conf_mask is of shape (B, H, W)
    valid_indices = conf_mask  # Boolean mask

    # Flatten the valid points
    pts3d_valid = pts3d[valid_indices]      # Shape: (N, 3)
    pixels_valid = pixels[valid_indices]    # Shape: (N, 2)

    if pts3d_valid.numel() == 0:
        # No valid points, return a default focal length
        focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
        return torch.tensor([focal_base])

    if focal_mode == "median":
        with torch.no_grad():
            # Direct estimation of focal
            u, v = pixels_valid.unbind(dim=-1)
            x, y, z = pts3d_valid.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # Assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(-1), fy_votes.view(-1)), dim=-1)
            focal = torch.nanmedian(f_votes).unsqueeze(0)  # Shape: (1,)

    elif focal_mode == "weiszfeld":
        # Initialize focal with L2 closed-form solution
        xy_over_z = (pts3d_valid[..., :2] / pts3d_valid[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )  # Shape: (N, 2)

        dot_xy_px = (xy_over_z * pixels_valid).sum(dim=-1)  # Shape: (N,)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)          # Shape: (N,)

        focal = dot_xy_px.mean() / dot_xy_xy.mean()         # Shape: scalar

        # Iterative re-weighted least-squares
        for _ in range(100):
            # Re-weighting by inverse of distance
            dis = (pixels_valid - focal * xy_over_z).norm(dim=-1)  # Shape: (N,)
            w = dis.clip(min=1e-8).reciprocal()                    # Shape: (N,)
            # Update the scaling with the new weights
            focal = (w * dot_xy_px).sum() / (w * dot_xy_xy).sum()
        focal = focal.unsqueeze(0)  # Shape: (1,)
    else:
        raise ValueError(f"bad focal_mode={focal_mode}")

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    return focal
