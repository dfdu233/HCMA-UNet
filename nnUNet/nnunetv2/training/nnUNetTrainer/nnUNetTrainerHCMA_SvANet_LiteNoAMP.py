import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHCMA_SvANet_Lite import (
    nnUNetTrainerHCMA_SvANet_Lite,
)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerHCMA_SvANet_LiteNoAMP(nnUNetTrainerHCMA_SvANet_Lite):
    """Lite trainer variant that runs in pure FP32 (no AMP/autocast)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Conservative defaults for numerical stability in FP32.
        self.initial_lr = float(os.environ.get("NNUNET_LITE_NOAMP_LR", "4e-5"))
        self.max_grad_norm = float(os.environ.get("NNUNET_LITE_NOAMP_MAX_GRAD_NORM", "4.0"))
        self.frloss_warmup_epochs = int(os.environ.get("NNUNET_LITE_NOAMP_FRLOSS_WARMUP", "8"))
        # Warm-start can be unstable if source/target heads differ; keep opt-in.
        self.svanet_init_from_hcma = os.environ.get("NNUNET_SVANET_INIT_FROM_HCMA", "0").lower() in (
            "1",
            "true",
            "t",
            "yes",
            "y",
        )

    def on_train_start(self):
        super().on_train_start()
        # Force disable AMP scaler even on CUDA.
        self.grad_scaler = None
        self.print_to_log_file("[NoAMP] AMP/autocast disabled; running pure FP32")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-5,
        )
        max_steps = int(self.num_epochs * self.num_iterations_per_epoch)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, max_steps)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

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

        if not torch.isfinite(l):
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(
                f"WARNING: non-finite loss encountered, skipping batch (count={self._nonfinite_batches})"
            )
            return {"loss": float(self._last_finite_loss)}

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
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            fea, output = self.network(data)
            self._validate_target_labels(target, output.shape[1])
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

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}
