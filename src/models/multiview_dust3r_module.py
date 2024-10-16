from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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
from src.dust3r.cloud_opt.init_im_poses import fast_pnp

from concurrent.futures import ThreadPoolExecutor

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class AccululatedSum(BaseAggregator):
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

        # for averaging loss across batches
        self.epoch_fraction = 0.0  # these are identical across all GPUs, so just use plain variable
        self.train_total_samples = AccululatedSum()  # these need to be reduced across GPUs, so use Metric
        self.train_total_images = AccululatedSum()  # these need to be reduced across GPUs, so use Metric

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

    def forward(self, views: List[Dict[str, torch.Tensor]]) -> Any:
        return self.net(views)

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
        self.epoch_fraction = self.trainer.current_epoch + batch_idx / self.trainer.num_training_batches

        self.log("trainer/epoch", self.epoch_fraction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # log the details of the loss
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(f"train_detail/{key}", value, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of samples seen so far
        batch_size = views[0]["img"].shape[0]
        self.train_total_samples(batch_size)
        self.log("trainer/total_samples", self.train_total_samples.compute(), on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of images seen so far
        num_views = len(views)
        n_image_cur_step = batch_size * num_views
        self.train_total_images(n_image_cur_step)
        self.log("trainer/total_images", self.train_total_images.compute(), on_step=True, on_epoch=False, prog_bar=False)

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
        self.log(f"val/loss_{dataset_name}", loss, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

        # Log the details of the loss with dataset name and view number in the key
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(
                    f"val_detail/{dataset_name}_view_{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    reduce_fx="mean",
                    sync_dist=True,
                    add_dataloader_idx=False,
                    batch_size=batch_size,
                )

        # Evaluate metrics for camera poses
        if dataset_name == "Co3d_v2":
            self.evaluate_camera_poses(views, preds, estimate_focal_from_first_view=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss, prog_bar=True)

        # if we dont do these, wandb for some reason cannot display the validation loss with them as the x-axis
        self.log("trainer/epoch", self.epoch_fraction)
        self.log("trainer/total_samples", self.train_total_samples.compute())
        self.log("trainer/total_images", self.train_total_images.compute())

    # def test_step(
    #     self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    # ) -> None:
    #     pass

    def evaluate_camera_poses(self, views, preds, estimate_focal_from_first_view=False):
        """Evaluate camera poses and focal lengths using fast_pnp in parallel."""
        batch_size = views[0]["img"].shape[0]

        # Estimate camera poses using the provided function
        poses_c2w_estimated, estimated_focals = self.estimate_camera_poses(preds=preds, views=views, niter_PnP=10, device=self.device, estimate_focal_from_first_view=estimate_focal_from_first_view)

        # Get ground truth poses
        poses_c2w_gt = [view['camera_pose'] for view in views]

        # Convert poses to tensors
        device = self.device
        pred_cameras = torch.tensor(np.stack(poses_c2w_estimated), device=device)  # Shape (B, num_views, 4, 4)
        gt_cameras = torch.stack(poses_c2w_gt).transpose(0, 1)  # (B, num_views, 4, 4)

        # compute the metrics: RRA, RTA, mAA
        # Ensure we have enough poses to compute relative errors
        if pred_cameras.shape[1] >= 2:

            def process_sample(sample_idx):
                my_preds = preds
                my_views = views

                # Extract camera poses for the current sample
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
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_sample, range(batch_size)))

            # Update metrics for all samples in the batch
            for results in batch_results:
                for tau in self.RRA_thresholds:
                    getattr(self, f'val_RRA_{tau}')(results[f"RRA_at_{tau}"])
                    self.log(f"val/RRA_at_{tau}", getattr(self, f'val_RRA_{tau}'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
                for tau in self.RTA_thresholds:
                    getattr(self, f'val_RTA_{tau}')(results[f"RTA_at_{tau}"])
                    self.log(f"val/RTA_at_{tau}", getattr(self, f'val_RTA_{tau}'), on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)
                self.val_mAA(results['mAA_30'])
                self.log("val/mAA_30", self.val_mAA, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean", sync_dist=True, add_dataloader_idx=False, batch_size=batch_size)

        else:
            log.warning("Not enough camera poses to compute relative errors.")

    # Function to estimate camera poses using fast_pnp
    @staticmethod
    def estimate_camera_poses(preds, views=None, niter_PnP=10, device='cpu', estimate_focal_from_first_view=False):
        """Estimate camera poses and focal lengths using fast_pnp in parallel."""
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

        estimate_cam_pose_one_sample_partial = partial(estimate_cam_pose_one_sample, device=device, niter_PnP=niter_PnP)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(estimate_cam_pose_one_sample_partial, data_for_processing))

        # Collect results from all processed samples
        for poses_c2w_sample, estimated_focals_sample in results:
            poses_c2w_all.append(poses_c2w_sample)
            estimated_focals_all.append(estimated_focals_sample)

        return poses_c2w_all, estimated_focals_all

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
            self.net.load_from_dust3r_checkpoint(self.pretrained)

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
            device,
            pp=None,  # Use default principal point (center of image)
            niter_PnP=niter_PnP
        )

        if pose_c2w is None or focal_length is None:
            print(f"Failed to estimate pose for view {view_idx}")
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

