from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric, Metric
from torchmetrics.aggregation import BaseAggregator
from src.dust3r.model import FlashDUSt3R
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

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
        ckpt = torch.load(self.pretrained)

        if isinstance(self.net, FlashDUSt3R):  # if the model is FlashDUSt3R, use the weights of the first head only
            ckpt = self._update_ckpt_keys(ckpt, new_head_name='downstream_head', head_to_keep='downstream_head1', head_to_discard='downstream_head2')

        self.net.load_state_dict(ckpt["model"], strict=False)
        del ckpt  # in case it occupies memory

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
