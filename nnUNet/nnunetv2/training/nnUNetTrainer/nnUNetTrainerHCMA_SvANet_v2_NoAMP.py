import torch
import numpy as np
from torch import nn
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_SvANet_v2 import HCMA_SvANet_v2
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class nnUNetTrainerHCMA_SvANet_v2_NoAMP(nnUNetTrainer):
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
        self.initial_lr = 4e-4
        self.weight_decay = 5e-2
        self.max_grad_norm = 12.0
        self.frloss_warmup_epochs = 2
        
        # 记录状态
        self._last_finite_loss = 1.0
        self._nonfinite_batches = 0

    @staticmethod
    def _is_finite_tensor(x) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).all())
        if isinstance(x, (list, tuple)):
            return all(nnUNetTrainerHCMA_SvANet_v2_NoAMP._is_finite_tensor(i) for i in x)
        return True

    def _grads_are_finite(self) -> bool:
        for p in self.network.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True


    @staticmethod
    def _set_predict_mode_recursive(module: nn.Module, value: bool):
        if hasattr(module, 'predict_mode'):
            module.predict_mode = value
        for attr in ('module', '_orig_mod'):
            if hasattr(module, attr):
                inner = getattr(module, attr)
                if isinstance(inner, nn.Module) and hasattr(inner, 'predict_mode'):
                    inner.predict_mode = value    


    def perform_actual_validation(self, save_probabilities: bool = False):
        self._set_predict_mode_recursive(self.network, True)
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self._set_predict_mode_recursive(self.network, False)


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
        # 显式将 grad_scaler 设为 None 以禁用 nnUNet 默认的 AMP 逻辑
        self.grad_scaler = None
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._nonfinite_batches = 0
        if self.current_epoch < self.frloss_warmup_epochs:
            self.print_to_log_file(f"FRLoss warmup: CE+Dice only (Epoch {self.current_epoch})")

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # --- 直接进行 FP32 前向传播 (移除 autocast) ---
        fea, output = self.network(data)

        # 稳定性检查：检查模型输出是否包含 NaN
        if (not self._is_finite_tensor(fea)) or (not self._is_finite_tensor(output)):
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(f"WARNING: NaN in model output, skipping (count={self._nonfinite_batches})")
            return {"loss": float(self._last_finite_loss)}

        feature_for_loss = None if self.current_epoch < self.frloss_warmup_epochs else fea
        l = self.loss(output, target, feature_for_loss)

        # 稳定性检查：检查 Loss 是否有限
        if not torch.isfinite(l):
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(f"WARNING: NaN loss, skipping (count={self._nonfinite_batches})")
            return {"loss": float(self._last_finite_loss)}

        # --- 标准反向传播 (不使用 grad_scaler) ---
        l.backward()

        # 稳定性检查：检查梯度是否爆炸
        if not self._grads_are_finite():
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(f"WARNING: NaN gradients, skipping step (count={self._nonfinite_batches})")
            return {"loss": float(self._last_finite_loss)}

        # 梯度裁剪及参数更新
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        self._last_finite_loss = float(l.detach().cpu().item())
        return {"loss": self._last_finite_loss}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # 验证阶段同样移除 autocast
        with torch.no_grad():
            fea, output = self.network(data)
            
            # 处理 NaN 增强鲁棒性
            if not self._is_finite_tensor(output):
                output = torch.nan_to_num(output, nan=0.0)

            feature_for_loss = None if self.current_epoch < self.frloss_warmup_epochs else fea
            l = self.loss(output, target, feature_for_loss)

        # 分割指标计算逻辑
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

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes)

        return {
            'loss': l.detach().cpu().numpy(), 
            'tp_hard': tp.detach().cpu().numpy()[1:] if not self.label_manager.has_regions else tp.detach().cpu().numpy(), 
            'fp_hard': fp.detach().cpu().numpy()[1:] if not self.label_manager.has_regions else fp.detach().cpu().numpy(), 
            'fn_hard': fn.detach().cpu().numpy()[1:] if not self.label_manager.has_regions else fn.detach().cpu().numpy()
        }

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        # 直接初始化 HCMA_SvANet_v2
        input_size = self.configuration_manager.patch_size
        model = HCMA_SvANet_v2(
            in_channels=num_input_channels, 
            n_classes=num_output_channels, 
            input_size=input_size, 
            predict_mode=False
        )
        return model