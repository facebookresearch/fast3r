# Built on top of https://github.com/HengyiWang/spann3r/blob/main/spann3r/tools/eval_recon.py

import numpy as np
from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
# import faiss

def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points, workers=24)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None, device=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=24)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None, device=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=24)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median

# def accuracy_faiss(gt_points, rec_points, gt_normals=None, rec_normals=None, device=None):
#     # Set up the Faiss index on GPU for the ground truth points
#     d = gt_points.shape[1]  # Dimension of the points
#     index = faiss.IndexFlatL2(d)  # L2 distance index
#     # index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device.index, index)

#     # Add ground truth points to the index
#     index.add(gt_points.astype(np.float32))

#     # Perform nearest neighbor search from reconstructed points to ground truth points
#     distances, idx = index.search(rec_points.astype(np.float32), 1)
#     acc = np.mean(distances)
#     acc_median = np.median(distances)

#     # Normal alignment calculations, if available
#     if gt_normals is not None and rec_normals is not None:
#         normal_dot = np.sum(gt_normals[idx.squeeze()] * rec_normals, axis=-1)
#         normal_dot = np.abs(normal_dot)
#         return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

#     return acc, acc_median

# def completion_faiss(gt_points, rec_points, gt_normals=None, rec_normals=None, device=None):
#     # Set up the Faiss index on GPU for the reconstructed points
#     d = rec_points.shape[1]  # Dimension of the points
#     index = faiss.IndexFlatL2(d)  # L2 distance index
#     # index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device.index, index)

#     # Add reconstructed points to the index
#     index.add(rec_points.astype(np.float32))

#     # Perform nearest neighbor search from ground truth points to reconstructed points
#     distances, idx = index.search(gt_points.astype(np.float32), 1)
#     comp = np.mean(distances)
#     comp_median = np.median(distances)

#     # Normal alignment calculations, if available
#     if gt_normals is not None and rec_normals is not None:
#         normal_dot = np.sum(gt_normals * rec_normals[idx.squeeze()], axis=-1)
#         normal_dot = np.abs(normal_dot)
#         return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

#     return comp, comp_median


def downsample_point_cloud(points, thresh):
    """
    Downsamples the point cloud so that no two points are closer than 'thresh'.
    Returns the downsampled points and indices of the points that were kept.
    """
    # Randomly shuffle the points to avoid bias
    rng = np.random.default_rng()
    indices = np.arange(points.shape[0])
    rng.shuffle(indices)
    points_shuffled = points[indices]

    # Fit NearestNeighbors with radius
    nn_engine = NearestNeighbors(radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(points_shuffled)
    rnn_idxs = nn_engine.radius_neighbors(points_shuffled, return_distance=False)

    # Create mask to keep only one point within each 'thresh' neighborhood
    mask = np.ones(points_shuffled.shape[0], dtype=bool)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = False
            mask[curr] = True
    downsampled_points = points_shuffled[mask]
    kept_indices = indices[mask]
    return downsampled_points, kept_indices


def accuracy_fast(gt_points, rec_points, gt_normals=None, rec_normals=None):
    # Parameters for optimization
    thresh = 0.01   # Adjust based on your dataset scale
    max_dist = 0.1  # Maximum distance to consider in metric computation

    # Downsample the reconstructed points and get indices
    rec_points_down, rec_downsample_indices = downsample_point_cloud(rec_points, thresh)
    if rec_normals is not None:
        rec_normals_down = rec_normals[rec_downsample_indices]

    # Build NearestNeighbors index on ground truth points
    nn_engine = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(gt_points)

    # Query nearest neighbors
    distances, idx = nn_engine.kneighbors(rec_points_down, return_distance=True)
    distances = distances.ravel()
    idx = idx.ravel()

    # Limit to maximum distance
    valid_mask = distances < max_dist
    distances = distances[valid_mask]
    idx = idx[valid_mask]
    rec_points_valid = rec_points_down[valid_mask]
    if rec_normals is not None:
        rec_normals_valid = rec_normals_down[valid_mask]

    # Compute mean and median accuracy
    acc = np.mean(distances) if distances.size > 0 else 0.0
    acc_median = np.median(distances) if distances.size > 0 else 0.0

    if gt_normals is not None and rec_normals is not None:
        gt_normals_matched = gt_normals[idx]
        normal_dot = np.sum(gt_normals_matched * rec_normals_valid, axis=-1)
        normal_dot = np.abs(normal_dot)
        nc = np.mean(normal_dot) if normal_dot.size > 0 else 0.0
        nc_median = np.median(normal_dot) if normal_dot.size > 0 else 0.0
        return acc, acc_median, nc, nc_median

    return acc, acc_median


def completion_fast(gt_points, rec_points, gt_normals=None, rec_normals=None):
    # Parameters for optimization
    thresh = 0.01   # Adjust based on your dataset scale
    max_dist = 0.1  # Maximum distance to consider in metric computation

    # Downsample the ground truth points and get indices
    gt_points_down, gt_downsample_indices = downsample_point_cloud(gt_points, thresh)
    if gt_normals is not None:
        gt_normals_down = gt_normals[gt_downsample_indices]

    # Build NearestNeighbors index on reconstructed points
    nn_engine = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(rec_points)

    # Query nearest neighbors
    distances, idx = nn_engine.kneighbors(gt_points_down, return_distance=True)
    distances = distances.ravel()
    idx = idx.ravel()

    # Limit to maximum distance
    valid_mask = distances < max_dist
    distances = distances[valid_mask]
    idx = idx[valid_mask]
    gt_points_valid = gt_points_down[valid_mask]
    if gt_normals is not None:
        gt_normals_valid = gt_normals_down[valid_mask]

    # Compute mean and median completion
    comp = np.mean(distances) if distances.size > 0 else 0.0
    comp_median = np.median(distances) if distances.size > 0 else 0.0

    if gt_normals is not None and rec_normals is not None:
        rec_normals_matched = rec_normals[idx]
        gt_normals_valid = gt_normals_valid  # Already downsampled and masked
        normal_dot = np.sum(gt_normals_valid * rec_normals_matched, axis=-1)
        normal_dot = np.abs(normal_dot)
        nc = np.mean(normal_dot) if normal_dot.size > 0 else 0.0
        nc_median = np.median(normal_dot) if normal_dot.size > 0 else 0.0
        return comp, comp_median, nc, nc_median

    return comp, comp_median




def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou
