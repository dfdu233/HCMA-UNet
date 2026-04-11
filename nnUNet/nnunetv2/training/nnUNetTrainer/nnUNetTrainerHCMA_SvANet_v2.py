import torch
from torch import autocast
import numpy as np
import os
import glob
from torch import nn
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_SvANet_v2 import HCMA_SvANet_v2
from torch.cuda.amp import autocast as dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.evaluation.evaluate_predictions import compute_metrics
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot

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
        self.frloss_warmup_epochs = 0
        self._last_finite_loss = 1.0
        self._nonfinite_batches = 0
        # Case-004 gate is optional for ablations; disabled by default to avoid
        # introducing extra bias/overhead in baseline SvANet training.
        self.case004_check_every = int(os.environ.get("NNUNET_CASE004_CHECK_EVERY", "0"))
        self.case004_dice_threshold = float(os.environ.get("NNUNET_CASE004_DICE_THRESHOLD", "0.7"))
        self.case004_key_preferred = "case_004"
        self._case004_reached_threshold = False
        self.svanet_init_from_hcma = os.environ.get("NNUNET_SVANET_INIT_FROM_HCMA", "1").lower() in (
            "1", "true", "t", "yes", "y"
        )
        self.svanet_init_hcma_ckpt = os.environ.get("NNUNET_HCMA_CKPT", "/root/workspace/nnUNet_results/Dataset666_Breast/nnUNetTrainerHCMA__nnUNetPlans__3d_fullres_v1/666_3d_fullres_nnUNetTrainerHCMA_fold0/fold_0/checkpoint_final.pth").strip()
        self._warmstarted_from_hcma = False

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

    def _resolve_hcma_checkpoint(self) -> str:
        if self.svanet_init_hcma_ckpt:
            return self.svanet_init_hcma_ckpt

        dataset_name = self.plans_manager.dataset_name
        fold_dir = f"fold_{self.fold}"
        pattern = os.path.join(
            os.environ.get("nnUNet_results", "/root/workspace/nnUNet_results"),
            dataset_name,
            "nnUNetTrainerHCMA__*",
            fold_dir,
            "checkpoint_final.pth",
        )
        candidates = sorted(glob.glob(pattern))
        return candidates[-1] if len(candidates) > 0 else ""

    def _try_warmstart_from_hcma(self):
        if not self.svanet_init_from_hcma or self._warmstarted_from_hcma:
            return
        if self.current_epoch != 0:
            return

        ckpt_path = self._resolve_hcma_checkpoint()
        if not ckpt_path or (not os.path.isfile(ckpt_path)):
            self.print_to_log_file("[warmstart] HCMA checkpoint not found, training from scratch")
            self._warmstarted_from_hcma = True
            return

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        src = checkpoint.get("network_weights", checkpoint)

        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod

        dst_state = mod.state_dict()
        compatible = {}
        for k, v in src.items():
            kk = k[7:] if (k.startswith("module.") and k[7:] in dst_state) else k
            if kk in dst_state and tuple(dst_state[kk].shape) == tuple(v.shape):
                compatible[kk] = v

        mod.load_state_dict(compatible, strict=False)
        self.print_to_log_file(
            f"[warmstart] initialized from HCMA checkpoint: {ckpt_path}; loaded {len(compatible)}/{len(dst_state)} tensors"
        )
        self._warmstarted_from_hcma = True

    def on_train_start(self):
        super().on_train_start()
        self._try_warmstart_from_hcma()

    def _find_case004_key(self):
        _, val_keys = self.do_split()
        if self.case004_key_preferred in val_keys:
            return self.case004_key_preferred
        for candidate in ("case_04", "Case_004", "Case_04"):
            if candidate in val_keys:
                return candidate
        for k in val_keys:
            if "004" in k or k.endswith("_04"):
                return k
        return None

    def _run_single_case_validation(self, case_key: str) -> float:
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        self._set_predict_mode_recursive(self.network, True)
        try:
            tile_step_size = float(os.environ.get('NNUNET_VAL_TILE_STEP_SIZE', '0.5'))
            use_gaussian = os.environ.get('NNUNET_VAL_USE_GAUSSIAN', '1').lower() in ('1', 'true', 't', 'yes', 'y')
            use_mirroring = os.environ.get('NNUNET_VAL_USE_MIRRORING', '1').lower() in ('1', 'true', 't', 'yes', 'y')
            predictor = nnUNetPredictor(
                tile_step_size=tile_step_size,
                use_gaussian=use_gaussian,
                use_mirroring=use_mirroring,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            predictor.manual_initialization(
                self.network,
                self.plans_manager,
                self.configuration_manager,
                None,
                self.dataset_json,
                self.__class__.__name__,
                self.inference_allowed_mirroring_axes,
            )

            dataset_val = nnUNetDataset(
                self.preprocessed_dataset_folder,
                [case_key],
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                num_images_properties_loading_threshold=0,
            )
            data, seg, properties = dataset_val.load_case(case_key)
            if self.is_cascaded:
                data = np.vstack((
                    data,
                    convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels, output_dtype=data.dtype),
                ))
            data = torch.from_numpy(np.ascontiguousarray(data.copy()))

            pred_logits = predictor.predict_sliding_window_return_logits(data).cpu()
            quick_val_folder = join(self.output_folder, 'validation_case004')
            maybe_mkdir_p(quick_val_folder)
            pred_file_truncated = join(quick_val_folder, case_key)
            export_prediction_from_logits(
                pred_logits,
                properties,
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                pred_file_truncated,
                False,
            )

            gt_file = join(self.preprocessed_dataset_folder_base, 'gt_segmentations', case_key + self.dataset_json['file_ending'])
            pred_file = pred_file_truncated + self.dataset_json['file_ending']
            labels = self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels
            metric = compute_metrics(
                gt_file,
                pred_file,
                self.plans_manager.image_reader_writer_class(),
                labels,
                self.label_manager.ignore_label,
            )
            case_dice = float(metric['metrics'][labels[0]]['Dice'])
            self.print_to_log_file(
                f"[case004-check] epoch={self.current_epoch} case={case_key} dice={case_dice:.4f} threshold={self.case004_dice_threshold:.2f}",
                also_print_to_console=True,
            )
            return case_dice
        finally:
            self._set_predict_mode_recursive(self.network, False)
            self.set_deep_supervision_enabled(True)

    def _maybe_run_case004_gate(self):
        if self._case004_reached_threshold:
            return
        if self.case004_check_every <= 0:
            return
        if self.current_epoch <= 0 or (self.current_epoch % self.case004_check_every) != 0:
            return

        case_key = self._find_case004_key()
        if case_key is None:
            self.print_to_log_file("[case004-check] case_004 not found in validation split, skip gate check")
            return

        case_dice = self._run_single_case_validation(case_key)
        if case_dice >= self.case004_dice_threshold:
            self._case004_reached_threshold = True
            self.print_to_log_file(
                f"[case004-check] Dice reached {case_dice:.4f} >= {self.case004_dice_threshold:.2f}, stop training and run full validation",
                also_print_to_console=True,
            )
            self.perform_actual_validation(save_probabilities=False)

    def run_training(self):
        self.on_train_start()

        while self.current_epoch < self.num_epochs and not self._case004_reached_threshold:
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            self.on_epoch_end()
            self._maybe_run_case004_gate()

        self.on_train_end()
