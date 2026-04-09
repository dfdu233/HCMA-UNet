import torch
from torch import autocast
import numpy as np
from torch import nn
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_SvANet_v2 import HCMA_SvANet_v2
from torch.cuda.amp import autocast as dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class nnUNetTrainerHCMA_SvANet_v2(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        exp_name: str = 'default',
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, exp_name, device)
        self.enable_deep_supervision = False
        self.num_epochs = 100
        self.oversample_foreground_percent = 0.33
        self.batch_size = 2
        self.initial_lr = 4e-4
        self.weight_decay = 5e-2
        self.num_iterations_per_epoch = 200
        self.max_grad_norm = 12.0
        self.frloss_warmup_epochs = 2
        self._last_finite_loss = 1.0
        self._nonfinite_batches = 0

    @staticmethod
    def _is_finite_tensor(x) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).all())
        if isinstance(x, (list, tuple)):
            return all(nnUNetTrainerHCMA_SvANet_v2._is_finite_tensor(i) for i in x)
        return True

    def _grads_are_finite(self) -> bool:
        for p in self.network.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.initial_lr,
            epochs=self.num_epochs,
            pct_start=0.06,
            steps_per_epoch=self.num_iterations_per_epoch,
            anneal_strategy="linear",
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self._nonfinite_batches = 0
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=8)}"
        )
        if self.current_epoch < self.frloss_warmup_epochs:
            self.print_to_log_file(
                f"FRLoss warmup active (epoch {self.current_epoch}/{self.frloss_warmup_epochs - 1}), using CE+Dice only"
            )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            fea, output = self.network(data)
            if (not self._is_finite_tensor(fea)) or (not self._is_finite_tensor(output)):
                self.optimizer.zero_grad(set_to_none=True)
                self._nonfinite_batches += 1
                self.print_to_log_file(
                    f"WARNING: non-finite feature/logit encountered, skipping batch (count={self._nonfinite_batches})"
                )
                return {"loss": float(self._last_finite_loss)}
            self._validate_target_labels(target, output.shape[1])
            feature_for_loss = None if self.current_epoch < self.frloss_warmup_epochs else fea
            l = self.loss(output, target, feature_for_loss)

        # Guard against non-finite loss to prevent optimizer state corruption.
        if not torch.isfinite(l):
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(
                f"WARNING: non-finite loss encountered, skipping batch (count={self._nonfinite_batches})"
            )
            return {"loss": float(self._last_finite_loss)}

        if self.grad_scaler is not None:
            prev_scale = self.grad_scaler.get_scale()
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            if not self._grads_are_finite():
                self.optimizer.zero_grad(set_to_none=True)
                # Keep GradScaler state machine consistent when we skip stepping after unscale_.
                self.grad_scaler.update()
                self._nonfinite_batches += 1
                self.print_to_log_file(
                    f"WARNING: non-finite gradients detected, skipping optimizer step (count={self._nonfinite_batches})"
                )
                return {"loss": float(self._last_finite_loss)}
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # Only advance scheduler if optimizer step was effectively executed.
            if self.grad_scaler.get_scale() >= prev_scale:
                self.lr_scheduler.step()
            else:
                self.print_to_log_file("GradScaler overflow detected, skipping lr_scheduler.step this iteration")
        else:
            l.backward()
            if not self._grads_are_finite():
                self.optimizer.zero_grad(set_to_none=True)
                self._nonfinite_batches += 1
                self.print_to_log_file(
                    f"WARNING: non-finite gradients detected, skipping optimizer step (count={self._nonfinite_batches})"
                )
                return {"loss": float(self._last_finite_loss)}
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()

        self._last_finite_loss = float(l.detach().cpu().item())
        return {"loss": self._last_finite_loss}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            fea, output = self.network(data)
            self._validate_target_labels(target, output.shape[1])
            del data
            feature_for_loss = None if self.current_epoch < self.frloss_warmup_epochs else fea
            l = self.loss(output, target, feature_for_loss)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        input_size = self.configuration_manager.patch_size
        model = HCMA_SvANet_v2(num_input_channels, num_output_channels, input_size=input_size, predict_mode=False)
        return model

    @staticmethod
    def _set_predict_mode_recursive(module: nn.Module, value: bool):
        if hasattr(module, 'predict_mode'):
            module.predict_mode = value
        for attr in ('module', '_orig_mod'):
            if hasattr(module, attr):
                inner = getattr(module, attr)
                if isinstance(inner, nn.Module) and hasattr(inner, 'predict_mode'):
                    inner.predict_mode = value

    def _validate_target_labels(self, target, num_classes: int):
        if self.label_manager.has_regions:
            return
        t = target[0] if isinstance(target, list) else target
        if t.dtype != torch.long:
            t = t.long()
        if self.label_manager.has_ignore_label:
            valid = t[t != self.label_manager.ignore_label]
        else:
            valid = t
        if valid.numel() == 0:
            return
        min_label = int(valid.min().item())
        max_label = int(valid.max().item())
        if min_label < 0 or max_label >= num_classes:
            raise RuntimeError(
                f"Target label out of range. min={min_label}, max={max_label}, num_classes={num_classes}."
            )

    def perform_actual_validation(self, save_probabilities: bool = False):
        self._set_predict_mode_recursive(self.network, True)
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self._set_predict_mode_recursive(self.network, False)
