import os
import torch
from torch import nn
import numpy as np
from typing import Union, Tuple, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainernn(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        exp_name: str = '',
        device: torch.device = torch.device('cuda'),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, exp_name, device)
        desired_bs = max(1, int(os.environ.get('NNUNET_NN_BATCH_SIZE', '2')))
        self.configuration_manager.configuration['batch_size'] = desired_bs
        self.enable_deep_supervision = False
        self.num_epochs = 200
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 200
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self._fg_log_train_batches = 3
        self._fg_log_val_batches = 3
        self._train_batch_idx = 0
        self._val_batch_idx = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        # Base trainer steps scheduler once per epoch, so use the nnUNet default
        # epoch-wise scheduler to avoid LR collapsing when switching optimizers.
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

    def _build_loss(self):
        self.enable_deep_supervision = False
        return super()._build_loss()

    def on_train_start(self):
        super().on_train_start()
        model = self.network.module if self.is_ddp and hasattr(self.network, "module") else self.network
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print_to_log_file(
            "[nn-arch] "
            f"model={model.__class__.__name__} params={num_params} trainable={trainable_params} "
            f"patch_size={tuple(self.configuration_manager.patch_size)} "
            f"batch_size={self.configuration_manager.batch_size} epochs={self.num_epochs} "
            f"iters_per_epoch={self.num_iterations_per_epoch} initial_lr={self.initial_lr}"
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._train_batch_idx = 0
        self._val_batch_idx = 0

    def _log_foreground_stats(self, output: torch.Tensor, target, phase: str, batch_idx: int):
        if output.ndim != 5 or output.shape[1] < 2:
            return
        t = target[0] if isinstance(target, list) else target
        if t.ndim == 5 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim == 5 and t.shape[1] > 1:
            t = t.argmax(1)
        if t.ndim != 4:
            return

        pred = output.argmax(1)
        pred_fg_ratio = float((pred > 0).float().mean().item())
        tgt_fg_ratio = float((t > 0).float().mean().item())
        fg_logit_mean = float(output[:, 1].mean().item())
        bg_logit_mean = float(output[:, 0].mean().item())
        self.print_to_log_file(
            f"[nn-fg-monitor][{phase}] epoch={self.current_epoch} batch={batch_idx} "
            f"pred_fg_ratio={pred_fg_ratio:.6f} target_fg_ratio={tgt_fg_ratio:.6f} "
            f"fg_logit_mean={fg_logit_mean:.4f} bg_logit_mean={bg_logit_mean:.4f}"
        )

    def train_step(self, batch: dict) -> dict:
        out = super().train_step(batch)
        if self._train_batch_idx < self._fg_log_train_batches:
            with torch.no_grad():
                data = batch['data'].to(self.device, non_blocking=True)
                target = batch['target']
                if isinstance(target, list):
                    target = [i.to(self.device, non_blocking=True) for i in target]
                else:
                    target = target.to(self.device, non_blocking=True)
                output = self.network(data)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                self._log_foreground_stats(output, target, 'train', self._train_batch_idx)
        self._train_batch_idx += 1
        return out

    def validation_step(self, batch: dict) -> dict:
        out = super().validation_step(batch)
        if self._val_batch_idx < self._fg_log_val_batches:
            with torch.no_grad():
                data = batch['data'].to(self.device, non_blocking=True)
                target = batch['target']
                if isinstance(target, list):
                    target = [i.to(self.device, non_blocking=True) for i in target]
                else:
                    target = target.to(self.device, non_blocking=True)
                output = self.network(data)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                self._log_foreground_stats(output, target, 'val', self._val_batch_idx)
        self._val_batch_idx += 1
        return out
