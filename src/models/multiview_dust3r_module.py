from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import re
import roma
import torch
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric, Metric
from torchmetrics.aggregation import BaseAggregator
from src.dust3r.post_process import estimate_focal_knowing_depth_and_confidence_mask
from src.dust3r.model import FlashDUSt3R
from src.models.fast3r import Fast3R
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.eval.cam_pose_metric import camera_to_rel_deg, calculate_auc
from src.eval.recon_metric import accuracy, accuracy_fast, completion, completion_fast
from src.dust3r.cloud_opt.init_im_poses import fast_pnp
import open3d as o3d
import time

from concurrent.futures import ThreadPoolExecutor

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class AccumulatedSum(BaseAggregator):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fn="sum",
            default_value=torch.tensor(0.0, dtype=torch.long),
            nan_strategy='warn',
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: int) -> None:
        self.sum_value += value

    def compute(self) -> torch.LongTensor:
        return self.sum_value


class MultiViewDUSt3RLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        train_criterion: torch.nn.Module,
        validation_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net', 'train_criterion', 'validation_criterion'])

        self.net = net
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        self.pretrained = pretrained
        self.resume_from_checkpoint = resume_from_checkpoint

        # use register_buffer to save these with checkpoints
        # so that when we resume training, these bookkeeping variables are preserved
        self.register_buffer("epoch_fraction", torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer("train_total_samples", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("train_total_images", torch.tensor(0, dtype=torch.long, device=self.device))

        self.train_total_samples_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric
        self.train_total_images_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric

        self.val_loss = MeanMetric()

        # Initialize metrics
        self.RRA_thresholds = [5, 15, 30]
        self.RTA_thresholds = [5, 15, 30]
        # Initialize RRA and RTA metrics as attributes
        for tau in self.RRA_thresholds:
            setattr(self, f'val_RRA_{tau}', MeanMetric())
        for tau in self.RTA_thresholds:
            setattr(self, f'val_RTA_{tau}', MeanMetric())

        self.val_mAA = MeanMetric()

        # Reconstruction evaluation metrics
        for dataset_name in ['dtu', '7scenes', 'nrgbd']:
            setattr(self, f'val_recon_{dataset_name}_accuracy', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_accuracy_median', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_completion', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_completion_median', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_nc1', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_nc1_median', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_nc2', MeanMetric())
            setattr(self, f'val_recon_{dataset_name}_nc2_median', MeanMetric())

    def forward(self, views: List[Dict[str, torch.Tensor]]) -> Any:
        return self.net(views)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Legacy: if the checkpoint does not contain the epoch_fraction, train_total_samples, and train_total_images
        # we manually add them to the checkpoint
        if checkpoint["state_dict"].get("epoch_fraction") is None:
            checkpoint["state_dict"]["epoch_fraction"] = self.epoch_fraction
        if checkpoint["state_dict"].get("train_total_samples") is None:
            checkpoint["state_dict"]["train_total_samples"] = self.train_total_samples
        if checkpoint["state_dict"].get("train_total_images") is None:
            checkpoint["state_dict"]["train_total_images"] = self.train_total_images

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        # the wandb logger lives in self.loggers
        # find the wandb logger and watch the model and gradients
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                # log gradients, parameter histogram and model topology
                self.wandb_logger.watch(self.net, log="all", log_freq=500, log_graph=False)

    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(self.trainer.train_dataloader.dataset, "set_epoch"):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(self.trainer.train_dataloader.sampler, "set_epoch"):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(self.current_epoch)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(self.current_epoch)

    def model_step(
        self, batch: List[Dict[str, torch.Tensor]], criterion: torch.nn.Module,
    ) -> Tuple[torch.Tensor, Dict]:
        device = self.device

        # Move data to device
        for view in batch:
            for name in "img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres".split():
                if name in view:
                    view[name] = view[name].to(device, non_blocking=True)

        views = batch

        preds = self.forward(views)

        # Compute the loss in higher precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            loss, loss_details = criterion(views, preds) if criterion is not None else None

        return views, preds, loss, loss_details

    def training_step(
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        views, preds, loss, loss_details = self.model_step(batch, self.train_criterion)
        self.epoch_fraction = torch.tensor(self.trainer.current_epoch + batch_idx / self.trainer.num_training_batches, device=self.device)

        self.log("trainer/epoch", self.epoch_fraction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # log the details of the loss
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(f"train_detail_{key}", value, on_step=True, on_epoch=False, prog_bar=False)
                match = re.search(r'/(\d{1,2})$', key)
                if match:
                    stripped_key = key[:match.start()]
                    self.log(f"train/{stripped_key}", value, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of samples seen so far
        batch_size = views[0]["img"].shape[0]
        self.train_total_samples_per_step(batch_size)  # aggregate across all GPUs
        self.train_total_samples += self.train_total_samples_per_step.compute()  # accumulate across all steps
        self.train_total_samples_per_step.reset()
        self.log("trainer/total_samples", self.train_total_samples, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of images seen so far
        num_views = len(views)
        n_image_cur_step = batch_size * num_views
        self.train_total_images_per_step(n_image_cur_step)  # aggregate across all GPUs
        self.train_total_images += self.train_total_images_per_step.compute()  # accumulate across all steps
        self.train_total_images_per_step.reset()
        self.log("trainer/total_images", self.train_total_images, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int, dataloader_idx: int,
    ) -> torch.Tensor:
        views, preds, loss, loss_details = self.model_step(batch, self.validation_criterion)

        # Extract the dataset name and batch size
        dataset_name = views[0]['dataset'][0]  # all views should have the same dataset name because we use "sequential" mode of CombinedLoader
        batch_size = views[0]["img"].shape[0]

        # Log the overall validation loss
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=True, batch_size=batch_size)
        self.val_loss(loss)
        # self.log(f"val/loss_{dataset_name}", loss, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

        # Log the details of the loss with dataset name and view number in the key
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(
                    f"val_detail_{dataset_name}_{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    reduce_fx="mean",
                    sync_dist=True,
                    add_dataloader_idx=False,
                    batch_size=batch_size,
                )
                match = re.search(r'/(\d{1,2})$', key)
                if match:
                    stripped_key = key[:match.start()]
                    self.log(f"val/{dataset_name}_{stripped_key}", value, on_step=False, on_epoch=True, prog_bar=False, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

        loss_value = loss.detach().cpu().item()
        del loss, loss_details
        torch.cuda.empty_cache()

        # Evaluate metrics for camera poses
        if dataset_name == "Co3d_v2":
            self.evaluate_camera_poses(views, preds, estimate_focal_from_first_view=True)

        # Evaluate point clouds only for the reconstruction datasets (DTU, 7-Scenes, and NRGBD)
        # eval only every 5 epochs because it's slow
        if dataset_name in ['dtu', '7scenes', 'nrgbd'] and (self.current_epoch % 5 == 4 or self.current_epoch == 0):
            self.evaluate_reconstruction(views, preds, dataset_name=dataset_name)

        del views, preds
        torch.cuda.empty_cache()

        return loss_value

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss, prog_bar=True)

        # if we dont do these, wandb for some reason cannot display the validation loss with them as the x-axis
        self.log("trainer/epoch", self.epoch_fraction)
        self.log("trainer/total_samples", self.train_total_samples)
        self.log("trainer/total_images", self.train_total_images)

    # def test_step(
    #     self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    # ) -> None:
    #     pass

    def evaluate_reconstruction(self, views, preds, dataset_name, min_conf_thr_percentile=60):
        # Correct orientation of the predicted points and confidence maps if necessary
        # self.correct_preds_orientation(preds, views)

        batch_size = len(views[0]['img'])  # Assuming batch_size is consistent

        # Initialize lists to collect per-sample metrics
        acc_list = []
        acc_med_list = []
        comp_list = []
        comp_med_list = []
        nc1_list = []
        nc1_med_list = []
        nc2_list = []
        nc2_med_list = []

        # Define the function to process a single sample
        def process_sample(i):
            pred_pts_list = []
            gt_pts_list_icp = []
            gt_pts_list_metrics = []
            colors_pred_list = []
            colors_gt_list = []
            conf_list = []
            weights_list = []

            for j, (view, pred) in enumerate(zip(views, preds)):
                # Extract predicted points and confidence
                pts_pred = pred['pts3d_in_other_view'][i]  # Shape: (H, W, 3)
                conf = pred['conf'][i]  # Shape: (H, W)

                ########### Debug: use GT point map to estimate poses
                # pts_gt = view['pts3d'][i]  # Shape: (H, W, 3)
                ########### End Debug

                # Extract GT points
                pts_gt = view['pts3d'][i]  # Shape: (H, W, 3)
                valid_mask = view['valid_mask'][i]  # Shape: (H, W)

                # Compute the confidence threshold for this view
                conf_flat = conf.view(-1)
                conf_threshold_value = torch.quantile(conf_flat, min_conf_thr_percentile / 100.0)

                # Create masks
                conf_mask = conf >= conf_threshold_value  # Shape: (H, W)
                final_mask_pred = valid_mask & conf_mask  # Predicted points: valid and high confidence
                final_mask_gt_icp = final_mask_pred       # GT points for ICP: same positions as high-confidence predicted points
                final_mask_gt_metrics = valid_mask        # GT points for metrics: all valid points

                # Apply masks to predicted points and conf
                pts_pred_masked = pts_pred[final_mask_pred]      # High-confidence predicted points
                conf_masked = conf[final_mask_pred]              # Corresponding confidence values

                # Apply mask to GT points for ICP
                pts_gt_masked_icp = pts_gt[final_mask_gt_icp]    # GT points corresponding to high-confidence predicted points

                # Apply mask to GT points for metrics
                pts_gt_masked_metrics = pts_gt[final_mask_gt_metrics]  # All valid GT points in this view
                gt_pts_list_metrics.append(pts_gt_masked_metrics)

                # Get image for colors
                img = view['img'][i]  # Shape: (3, H, W)
                img = img.permute(1, 2, 0)  # Shape: (H, W, 3)
                img = (img + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
                colors_pred_masked = img[final_mask_pred]  # Colors at high-confidence predicted points
                colors_gt_masked = img[final_mask_gt_metrics]  # Colors at all valid GT points

                # Weights for ICP alignment (all ones since we've already filtered low-confidence points)
                weights_masked = torch.ones_like(conf_masked, dtype=torch.float32)

                # Append to lists
                pred_pts_list.append(pts_pred_masked)
                gt_pts_list_icp.append(pts_gt_masked_icp)
                conf_list.append(conf_masked)
                colors_pred_list.append(colors_pred_masked)
                colors_gt_list.append(colors_gt_masked)
                weights_list.append(weights_masked)

            # Concatenate points, colors, confidences, and weights
            if len(pred_pts_list) == 0 or len(gt_pts_list_metrics) == 0:
                # If no valid points, return default metrics
                print(f"Sample {i}: No valid points found.")
                return 0, 0, 0, 0, 0, 0, 0, 0

            pred_pts_all = torch.cat(pred_pts_list, dim=0)           # Shape: (N_pred, 3)
            gt_pts_all_icp = torch.cat(gt_pts_list_icp, dim=0)       # Shape: (N_pred, 3)
            gt_pts_all_metrics = torch.cat(gt_pts_list_metrics, dim=0)  # Shape: (N_gt, 3)
            colors_pred_all = torch.cat(colors_pred_list, dim=0)               # Shape: (N_pred, 3)
            colors_gt_all = torch.cat(colors_gt_list, dim=0)               # Shape: (N_gt, 3)
            conf_all = torch.cat(conf_list, dim=0)                   # Shape: (N_pred,)
            weights_all = torch.cat(weights_list, dim=0)             # Shape: (N_pred,)

            # Ensure that the data is on CPU for Open3D and numpy operations
            pred_pts_tensor = pred_pts_all.cpu()          # Shape: (N_pred, 3)
            gt_pts_tensor_icp = gt_pts_all_icp.cpu()      # Shape: (N_pred, 3)
            gt_pts_tensor_metrics = gt_pts_all_metrics.cpu()  # Shape: (N_gt, 3)
            colors_pred_tensor = colors_pred_all.cpu()              # Shape: (N_pred, 3)
            colors_gt_tensor = colors_gt_all.cpu()              # Shape: (N_gt, 3)
            conf_tensor = conf_all.cpu()                  # Shape: (N_pred,)
            weights = weights_all.cpu()                   # Shape: (N_pred,)

            print(f"Sample {i}: Number of high-confidence predicted points: {pred_pts_tensor.shape[0]}")
            print(f"Sample {i}: Number of GT points for ICP: {gt_pts_tensor_icp.shape[0]}")
            print(f"Sample {i}: Number of GT points for metrics: {gt_pts_tensor_metrics.shape[0]}")

            # Align predicted points to GT using roma.rigid_points_registration with weights
            start_time = time.time()

            # Input tensors for ICP alignment (must have the same shape)
            x = pred_pts_tensor          # High-confidence predicted points (N_pred, 3)
            y = gt_pts_tensor_icp        # Corresponding GT points (N_pred, 3)

            # Compute the rigid transformation with scaling and weights
            R, t, s = roma.rigid_points_registration(x, y, weights=weights, compute_scaling=True)

            alignment_time = time.time() - start_time
            print(f"Alignment time using roma with weights for sample {i}: {alignment_time:.4f} seconds")

            # Apply the transformation to the predicted points
            pred_aligned = s * (x @ R.T) + t  # Shape: (N_pred, 3)

            # Estimate normals
            start_time = time.time()
            # Create point clouds in Open3D for normal estimation

            # Predicted point cloud (high-confidence points)
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_aligned.numpy())
            pred_pcd.colors = o3d.utility.Vector3dVector(colors_pred_tensor.numpy())
            pred_pcd.estimate_normals()

            # Ground truth point cloud for metrics (all valid points)
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_pts_tensor_metrics.numpy())
            gt_pcd.colors = o3d.utility.Vector3dVector(colors_gt_tensor.numpy())
            gt_pcd.estimate_normals()
            normals_time = time.time() - start_time
            print(f"Normal estimation time for sample {i}: {normals_time:.4f} seconds")

            # Get normals
            pred_normals = np.asarray(pred_pcd.normals)
            gt_normals = np.asarray(gt_pcd.normals)

            # Convert point clouds to numpy arrays for evaluation
            pred_points_np = np.asarray(pred_pcd.points)
            gt_points_np = np.asarray(gt_pcd.points)

            # Save the GT and predicted point clouds (separately) for visualization
            # Define file paths
            gt_pcd_path = f"gt_pcd_sample_{i}.ply"
            pred_pcd_path = f"pred_pcd_sample_{i}.ply"
            # Save the GT point cloud
            o3d.io.write_point_cloud(gt_pcd_path, gt_pcd)
            # Save the predicted point cloud
            o3d.io.write_point_cloud(pred_pcd_path, pred_pcd)

            # Compute metrics
            start_time = time.time()
            acc, acc_med, nc1, nc1_med = accuracy(
                gt_points_np, pred_points_np, gt_normals, pred_normals
            )
            comp, comp_med, nc2, nc2_med = completion(
                gt_points_np, pred_points_np, gt_normals, pred_normals
            )
            metrics_time = time.time() - start_time
            print(f"Metrics computation time for sample {i}: {metrics_time:.4f} seconds")
            print(f"Accuracy: {acc:.4f}, Accuracy median: {acc_med:.4f}")
            print(f"Completion: {comp:.4f}, Completion median: {comp_med:.4f}")
            print(f"Normal consistency 1: {nc1:.4f}, Normal consistency 1 median: {nc1_med:.4f}")
            print(f"Normal consistency 2: {nc2:.4f}, Normal consistency 2 median: {nc2_med:.4f}")

            # Return metrics
            return acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med

        # Use ThreadPoolExecutor to parallelize across samples
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_sample, i) for i in range(batch_size)]
            for future in futures:
                acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med = future.result()
                # Collect metrics
                acc_list.append(acc)
                acc_med_list.append(acc_med)
                comp_list.append(comp)
                comp_med_list.append(comp_med)
                nc1_list.append(nc1)
                nc1_med_list.append(nc1_med)
                nc2_list.append(nc2)
                nc2_med_list.append(nc2_med)

        # After processing all samples in batch, compute mean and median metrics
        mean_accuracy = np.mean(acc_list)
        median_accuracy = np.median(acc_list)
        mean_completion = np.mean(comp_list)
        median_completion = np.median(comp_list)
        mean_nc1 = np.mean(nc1_list)
        median_nc1 = np.median(nc1_list)
        mean_nc2 = np.mean(nc2_list)
        median_nc2 = np.median(nc2_list)

        # Update and log metrics
        getattr(self, f'val_recon_{dataset_name}_accuracy')(mean_accuracy)
        self.log(f"val_recon_{dataset_name}/accuracy", getattr(self, f'val_recon_{dataset_name}_accuracy'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_accuracy_median')(median_accuracy)
        self.log(f"val_recon_{dataset_name}/accuracy_median", getattr(self, f'val_recon_{dataset_name}_accuracy_median'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_completion')(mean_completion)
        self.log(f"val_recon_{dataset_name}/completion", getattr(self, f'val_recon_{dataset_name}_completion'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_completion_median')(median_completion)
        self.log(f"val_recon_{dataset_name}/completion_median", getattr(self, f'val_recon_{dataset_name}_completion_median'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_nc1')(mean_nc1)
        self.log(f"val_recon_{dataset_name}/nc1", getattr(self, f'val_recon_{dataset_name}_nc1'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_nc1_median')(median_nc1)
        self.log(f"val_recon_{dataset_name}/nc1_median", getattr(self, f'val_recon_{dataset_name}_nc1_median'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_nc2')(mean_nc2)
        self.log(f"val_recon_{dataset_name}/nc2", getattr(self, f'val_recon_{dataset_name}_nc2'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
        getattr(self, f'val_recon_{dataset_name}_nc2_median')(median_nc2)
        self.log(f"val_recon_{dataset_name}/nc2_median", getattr(self, f'val_recon_{dataset_name}_nc2_median'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

    def evaluate_camera_poses(self, views, preds, estimate_focal_from_first_view=False):
        """Evaluate camera poses and focal lengths using fast_pnp in parallel."""

        # in-place correction of the orientation of the predicted points and confidence maps
        # this is because the data loader transposed the input images and valid_masks to landscape
        self.correct_preds_orientation(preds, views)

        # Estimate camera poses using the provided function
        poses_c2w_estimated, estimated_focals = self.estimate_camera_poses(preds=preds, views=views, niter_PnP=10, estimate_focal_from_first_view=estimate_focal_from_first_view)

        # Get ground truth poses
        poses_c2w_gt = [view['camera_pose'] for view in views]

        # Convert poses to tensors
        device = self.device
        pred_cameras = torch.tensor(np.stack(poses_c2w_estimated), dtype=poses_c2w_gt[0].dtype, device=device)  # Shape (B, num_views, 4, 4)
        gt_cameras = torch.stack(poses_c2w_gt).transpose(0, 1)  # (B, num_views, 4, 4)

        # compute the metrics: RRA, RTA, mAA
        # Ensure we have enough poses to compute relative errors
        if pred_cameras.shape[1] >= 2:

            def process_sample(sample_idx):
                pred_sample = pred_cameras[sample_idx]  # Shape (num_views, 4, 4)
                gt_sample = gt_cameras[sample_idx]      # Shape (num_views, 4, 4)

                # Compute relative rotation and translation errors
                rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_sample, gt_sample, device, len(pred_sample))

                # Compute metrics for all tau thresholds
                results = {}
                for tau in self.RRA_thresholds:
                    results[f"RRA_at_{tau}"] = (rel_rangle_deg < tau).float().mean().item()
                for tau in self.RTA_thresholds:
                    results[f"RTA_at_{tau}"] = (rel_tangle_deg < tau).float().mean().item()

                # Compute mAA(30)
                results['mAA_30'] = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30).item()

                return results

            # Use ThreadPoolExecutor to process samples in parallel across the batch
            batch_size = views[0]["img"].shape[0]
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_sample, range(batch_size)))

            # Update metrics for all samples in the batch
            for results in batch_results:
                for tau in self.RRA_thresholds:
                    getattr(self, f'val_RRA_{tau}')(results[f"RRA_at_{tau}"])
                    self.log(f"val_metric/RRA_at_{tau}", getattr(self, f'val_RRA_{tau}'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
                for tau in self.RTA_thresholds:
                    getattr(self, f'val_RTA_{tau}')(results[f"RTA_at_{tau}"])
                    self.log(f"val_metric/RTA_at_{tau}", getattr(self, f'val_RTA_{tau}'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
                self.val_mAA(results['mAA_30'])
                self.log("val_metric/mAA_30", self.val_mAA, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

        else:
            log.warning("Not enough camera poses to compute relative errors.")

    # Function to estimate camera poses using fast_pnp
    @staticmethod
    def estimate_camera_poses(preds, views=None, niter_PnP=10, estimate_focal_from_first_view=False):
        """Estimate camera poses and focal lengths using fast_pnp in parallel."""

        batch_size = len(preds[0]["pts3d_in_other_view"])  # Get the batch size

        # Prepare data_for_processing
        data_for_processing = []

        for i in range(batch_size):
            # Collect preds for each sample in the batch
            sample_preds = [{key: value[i].cpu() for key, value in view.items()} for view in preds]

            data_for_processing.append(sample_preds)

        # Estimate the focal length
        def estimate_focal_for_sample(sample_preds):
            if estimate_focal_from_first_view:
                # Get the first view's pts3d and confidence map
                pts3d_i = sample_preds[0]["pts3d_in_other_view"].unsqueeze(0)  # Shape: (1, H, W, 3)
                conf_i = sample_preds[0]["conf"].unsqueeze(0)                  # Shape: (1, H, W)

                # Estimate focal length using the provided function and confidence mask
                estimated_focal = estimate_focal(pts3d_i, conf_i, min_conf_thr_percentile=90)

                # Store the estimated focal length in sample_preds
                for view_pred in sample_preds:
                    view_pred["focal_length"] = estimated_focal

            return sample_preds

        with ThreadPoolExecutor() as executor:
            data_for_processing = list(executor.map(estimate_focal_for_sample, data_for_processing))

        # Estimate the camera poses
        # Use ProcessPoolExecutor to parallelize processing across samples in the batch
        poses_c2w_all = []
        estimated_focals_all = []

        # Use partial to fix arguments
        from functools import partial

        estimate_cam_pose_one_sample_partial = partial(estimate_cam_pose_one_sample, niter_PnP=niter_PnP)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(estimate_cam_pose_one_sample_partial, data_for_processing))

        # Collect results from all processed samples
        for poses_c2w_sample, estimated_focals_sample in results:
            poses_c2w_all.append(poses_c2w_sample)
            estimated_focals_all.append(estimated_focals_sample)

        return poses_c2w_all, estimated_focals_all

    @staticmethod
    def correct_preds_orientation(preds, views):
        # *In-place* correction of the orientation of the predicted points and confidence maps

        # correct the shape of the predicted points and confidence maps if the view is portrait
        # this is because the data loader transposed the input images and valid_masks to landscape
        # see datasets/base/base_stereo_view_dataset.py
        if views is not None:
            for pred, view in zip(preds, views):
                # debug: use GT point map to estimate poses
                # pred["pts3d_in_other_view"] = view["pts3d"]  # shape (B, H, W, 3)
                # pred["conf"] = view['valid_mask'].float() if "valid_mask" in view else torch.ones_like(pred["conf"])  # shape (B, H, W)
                # pred["focal_length"] = view["camera_intrinsics"][:, 0, :2].sum(1)
                # end debug

                # check if the view is protrait or landscape (true_shape: (H, W))
                conf_list = []
                pts3d_list = []

                for i in range(view["true_shape"].shape[0]):
                    H, W = view["true_shape"][i]
                    if H > W:  # portrait
                        # Transpose the tensors
                        transposed_conf = pred["conf"][i].transpose(0, 1)
                        transposed_pts3d = pred["pts3d_in_other_view"][i].transpose(0, 1)

                        # Append the transposed tensors to the lists
                        conf_list.append(transposed_conf)
                        pts3d_list.append(transposed_pts3d)
                    else:
                        # Append the original tensors to the lists
                        conf_list.append(pred["conf"][i])
                        pts3d_list.append(pred["pts3d_in_other_view"][i])

                pred["conf"] = conf_list
                pred["pts3d_in_other_view"] = pts3d_list

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler

            # HACK: if the class is pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
            # both warmup_epochs and max_epochs should be scaled.
            # more specifically, max_epochs should be scaled to total number of steps that we will have during training,
            # and warmup_epochs should be scaled up proportionally.
            if scheduler_config.func is LinearWarmupCosineAnnealingLR:
                # Extract the keyword arguments from the partial object
                scheduler_kwargs = {k: v for k, v in scheduler_config.keywords.items()}
                original_warmup_epochs = scheduler_kwargs['warmup_epochs']
                original_max_epochs = scheduler_kwargs['max_epochs']

                total_steps = self.trainer.estimated_stepping_batches  # total number of total steps in all training epochs

                # Scale warmup_epochs and max_epochs
                scaled_warmup_epochs = int(original_warmup_epochs * total_steps / original_max_epochs)
                scaled_max_epochs = total_steps

                # Update the kwargs with scaled values
                scheduler_kwargs.update({
                    'warmup_epochs': scaled_warmup_epochs,
                    'max_epochs': scaled_max_epochs
                })

                # Re-initialize the scheduler with updated parameters
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    **scheduler_kwargs
                )
            else:
                scheduler = scheduler_config(optimizer=optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'name': 'train/lr',  # put lr inside train group in loggers
                    'scheduler': scheduler,
                    'interval': 'step' if scheduler_config.func is LinearWarmupCosineAnnealingLR else 'epoch',
                    'frequency': 1,
                }
            }

        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        # Load pretrained weights if available and not resuming
        # note that if resume_from_checkpoint is set, the Trainer is responsible for actually loading the checkpoint
        # so we are only using resume_from_checkpoint as a check of whether we should load the pretrained weights
        if self.pretrained and not self.resume_from_checkpoint:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        log.info(f"Loading pretrained: {self.pretrained}")
        if isinstance(self.net, FlashDUSt3R):  # if the model is FlashDUSt3R, use the weights of the first head only
            ckpt = torch.load(self.pretrained)
            ckpt = self._update_ckpt_keys(ckpt, new_head_name='downstream_head', head_to_keep='downstream_head1', head_to_discard='downstream_head2')
            self.net.load_state_dict(ckpt["model"], strict=False)
            del ckpt  # in case it occupies memory
        elif isinstance(self.net, Fast3R):
            if self.pretrained.endswith("DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"):
                # if the model is Fast3R and the pretrained model is DUSt3R, load a subset of the weights into the net
                self.net.load_from_dust3r_checkpoint(self.pretrained)
            else:
                # if the checkpoint is also Fast3R, load all weights
                log.info(f"Loading pretrained weights from {self.pretrained}")
                checkpoint = torch.load(self.pretrained)
                filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith('net.')}
                # Remove the 'net.' prefix from the keys
                filtered_state_dict = {k[len('net.'):]: v for k, v in filtered_state_dict.items()}
                # Load the filtered state_dict into the model
                self.net.load_state_dict(filtered_state_dict, strict=True)

    @staticmethod
    def _update_ckpt_keys(ckpt, new_head_name='downstream_head', head_to_keep='downstream_head1', head_to_discard='downstream_head2'):
        """Helper function to use the weights of a model with multiple heads in a model with a single head.
        specifically, keep only the weights of the first head and delete the weights of the second head.
        """
        new_ckpt = {'model': {}}

        for key, value in ckpt['model'].items():
            if key.startswith(head_to_keep):
                new_key = key.replace(head_to_keep, new_head_name)
                new_ckpt['model'][new_key] = value
            elif key.startswith(head_to_discard):
                continue
            else:
                new_ckpt['model'][key] = value

        return new_ckpt


def estimate_cam_pose_one_sample(sample_preds, device='cpu', niter_PnP=10):
    poses_c2w = []
    estimated_focals = []

    # Define the function to process each view
    def process_view(view_idx):
        pts3d = sample_preds[view_idx]["pts3d_in_other_view"].cpu().numpy().squeeze()  # (H, W, 3)
        valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() > 0.2  # Confidence mask
        focal_length = float(sample_preds[view_idx]["focal_length"]) if "focal_length" in sample_preds[view_idx] else None

        # Call fast_pnp with unflattened pts3d and mask
        focal_length, pose_c2w = fast_pnp(
            torch.tensor(pts3d),
            focal_length,  # Guess focal length
            torch.tensor(valid_mask, dtype=torch.bool),
            "cpu",
            pp=None,  # Use default principal point (center of image)
            niter_PnP=niter_PnP
        )

        if pose_c2w is None or focal_length is None:
            log.warning(f"Failed to estimate pose for view {view_idx}")
            return np.eye(4), focal_length  # Return identity pose in case of failure

        # Return the results for this view
        return pose_c2w.cpu().numpy(), focal_length

    # Use ThreadPoolExecutor to process views in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_view function to each view index
        results = list(executor.map(process_view, range(len(sample_preds))))

    # Collect the results
    for pose_c2w_result, focal_length_result in results:
        poses_c2w.append(pose_c2w_result)
        estimated_focals.append(focal_length_result)

    return poses_c2w, estimated_focals


def estimate_focal(pts3d_i, conf_i, pp=None, min_conf_thr_percentile=50):
    B, H, W, THREE = pts3d_i.shape
    assert B == 1  # Since we're processing one sample at a time

    if pp is None:
        pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device).view(1, 2)  # Shape: (1, 2)

    # Flatten the confidence map using reshape instead of view
    conf_flat = conf_i.reshape(-1)

    # Compute the confidence threshold based on the percentile
    percentile = min_conf_thr_percentile / 100.0  # Convert to a fraction
    conf_threshold = torch.quantile(conf_flat, percentile)

    # Create the confidence mask based on the computed threshold
    conf_mask = conf_i >= conf_threshold
    conf_mask = conf_mask.view(B, H, W)  # Ensure shape is (B, H, W)

    # Check if there are enough valid points
    if conf_mask.sum() < 10:  # Adjust the minimum number as needed
        print("Not enough high-confidence points for focal estimation.")
        # Optionally, adjust the percentile or set conf_mask to all True
        # For example:
        # conf_mask = torch.ones_like(conf_mask, dtype=torch.bool)

    focal = estimate_focal_knowing_depth_and_confidence_mask(
        pts3d_i, pp.unsqueeze(0), conf_mask, focal_mode="weiszfeld"
    ).ravel()
    return float(focal)
