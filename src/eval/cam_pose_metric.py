# This script is adapted from PoseDiffusion: https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/util/metric.py
# The only change is that the original script assumes PyTorch3D SE3 matrices, which is row-major, which assumes the translation vector is at the last row,
# while the script here assumes the standard column-major SE3 matrices that has the translation vector in the last column.

import numpy as np
import torch

from pytorch3d.transforms import so3_relative_angle


def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size):
    with torch.no_grad():
        # Convert cameras to 4x4 SE3 transformation matrices
        pred_se3 = pred_cameras  # (B, 4, 4)
        gt_se3 = gt_cameras  # (B, 4, 4)

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)
        # pair_idx_i2 = pair_idx_i2.to(device)

        # Compute relative camera poses between pairs
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])
        # relative_pose_gt = inv(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])  # these have numerical issues
        # relative_pose_pred = inv(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])  # these have numerical issues

        # Compute the difference in rotation and translation
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
        rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))


def calculate_auc(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B):
    N = B
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(1)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2

def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE(3) matrix in the batch.

    Args:
        se3 (Tensor): Nx4x4 tensor of SE(3) matrices.

    Returns:
        Tensor: Nx4x4 tensor of inverted SE(3) matrices.
    """
    # Extract rotation matrix R and translation vector t
    R = se3[:, :3, :3]            # Shape: (N, 3, 3)
    t = se3[:, :3, 3].unsqueeze(2)  # Shape: (N, 3, 1)

    # Compute the transpose (inverse) of the rotation matrix
    R_transposed = R.transpose(1, 2)  # Shape: (N, 3, 3)

    # Compute the new translation vector: -R^T * t
    t_inv = -torch.bmm(R_transposed, t)  # Shape: (N, 3, 1)

    # Construct the inverse SE(3) matrix
    inv_se3 = torch.zeros_like(se3)  # Initialize an empty tensor with the same shape

    # Set the rotation part
    inv_se3[:, :3, :3] = R_transposed

    # Set the translation part
    inv_se3[:, :3, 3] = t_inv.squeeze(2)

    # Set the bottom row to [0, 0, 0, 1]
    inv_se3[:, 3, 3] = 1.0

    return inv_se3



def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def compute_ARE(rotation1, rotation2):
    if isinstance(rotation1, torch.Tensor):
        rotation1 = rotation1.cpu().detach().numpy()
    if isinstance(rotation2, torch.Tensor):
        rotation2 = rotation2.cpu().detach().numpy()

    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    error = theta * 180 / np.pi
    return np.minimum(error, np.abs(180 - error))
