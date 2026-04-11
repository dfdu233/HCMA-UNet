import torch
from torch import autocast
import numpy as np
import os
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import HCMA
from torch.cuda.amp import autocast as dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.evaluation.evaluate_predictions import compute_metrics
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.training.loss.compound_losses import (
    DC_and_CE_loss,
    DC_and_BCE_loss,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

from typing import Union, Tuple, List
class nnUNetTrainerHCMA(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        exp_name='',
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, exp_name,device
        )
        quick_epochs = int(os.environ.get("NNUNET_QUICK_EPOCHS", "100"))
        self.num_epochs = 50 if quick_epochs <= 50 else 100
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 200
        self.batch_size = 2
        self.initial_lr = 1.5e-4
        self.weight_decay = 2e-2
        self.enable_deep_supervision = False  # Truse
        self._fg_log_train_batches = 3
        self._fg_log_val_batches = 3
        self._train_batch_idx = 0
        self._val_batch_idx = 0
        self.case004_check_every = 20
        self.case004_dice_threshold = 0.7
        self.case004_key_preferred = "case_004"
        self._case004_reached_threshold = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        # Use monotonic epoch-wise decay for stable continuation runs.
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self._train_batch_idx = 0
        self._val_batch_idx = 0
        # self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=8)}"
        )
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

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
            f"[fg-monitor][{phase}] epoch={self.current_epoch} batch={batch_idx} "
            f"pred_fg_ratio={pred_fg_ratio:.6f} target_fg_ratio={tgt_fg_ratio:.6f} "
            f"fg_logit_mean={fg_logit_mean:.4f} bg_logit_mean={bg_logit_mean:.4f}"
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            fea, output = self.network(data)
            self._validate_target_labels(target, output.shape[1])
            # Enable FRLoss path: loss(logits, target, feature)
            l = self.loss(output, target, fea)

        if self._train_batch_idx < self._fg_log_train_batches:
            self._log_foreground_stats(output.detach(), target, "train", self._train_batch_idx)
        self._train_batch_idx += 1

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            fea, output = self.network(data)
            self._validate_target_labels(target, output.shape[1])
            del data
            l = self.loss(output, target, fea)

        if self._val_batch_idx < self._fg_log_val_batches:
            self._log_foreground_stats(output.detach(), target, "val", self._val_batch_idx)
        self._val_batch_idx += 1

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
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
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)
        """
        # model = Baselinev5_DenseDown(in_channels=1,n_classes=2,predict_mode=True)
        # model = Baselinev5(
        #     num_input_channels,
        #     num_output_channels,
        #     deep_supervision=enable_deep_supervision,
        # )
        # model = FrigeSelfAxialMamba(num_input_channels,2,predict_mode=False)
        model = HCMA(
            num_input_channels,
            num_output_channels,
            patch_ini=list(self.configuration_manager.patch_size),
            predict_mode=False,
            use_small_lesion_refine=True,
        )
        # model = AxialMamba(num_input_channels,2,predict_mode=False)
        # model = SingleBaselinev5(num_input_channels,2,predict_mode=True)
        # model = SingleMamba(num_input_channels,2,predict_mode=True)
        # model = DifferentPatch(num_input_channels,2,predict_mode=False)
        # model = nnFormer(input_channels=num_input_channels,num_classes=num_output_channels)
        # model = Baselinev5_DenseDown(
        #     num_input_channels,
        #     num_output_channels,
        #     deep_supervision=enable_deep_supervision,
        # )



     


        # model = UNETR_PP(
        #         in_channels=1,
        #         out_channels=2,  # 假设分割为2类
        #         feature_size=16,
        #         hidden_size=256,
        #         num_heads=8,
        #         pos_embed="perceptron",
        #         norm_name="instance",
        #         dropout_rate=0.1,
        #         depths=[3, 3, 3, 3],
        #         dims=[32, 64, 128, 256,512],
        #         conv_op=nn.Conv3d,
        #         do_ds=False,
        #         predict_mode=True
        #     )

        # model=MedNeXt(
        # in_channels=num_input_channels,
        # n_channels=32,
        # n_classes=2,
        # exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        # kernel_size=3,
        # deep_supervision=False,
        # do_res=True,
        # do_res_up_down=True,
        # block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        # predict_mode=True
        # )


        
        # model = UNETR(num_input_channels, num_output_channels, img_size=(128, 128, 128))



        # model = SwinUNETR(
        #     img_size=(128, 128, 128),
        #     in_channels=num_input_channels,
        #     out_channels=num_output_channels,
        #     use_v2=False,
        #     predict_mode=True
        # )



        # model = AttentionUnet(
        #     3,
        #     in_channels=num_input_channels,
        #     out_channels=num_output_channels,
        #     channels=[32, 64, 128, 256, 512],
        #     strides=[2, 2, 2, 2],
        #     predict_mode=True
        # )

        # model = UXNET(
        #     num_input_channels,
        #     num_output_channels,
        #     # deep_supervision=enable_deep_supervision,
        #     predict_mode=True
        # )
        return model

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

    @staticmethod
    def _set_predict_mode_recursive(module: nn.Module, value: bool):
        """Set predict_mode for wrapped or nested modules when the attribute exists."""
        if hasattr(module, 'predict_mode'):
            module.predict_mode = value

        # Handle wrapped modules (DDP / compile wrappers) and nested modules.
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
                f"Target label out of range. min={min_label}, max={max_label}, "
                f"num_classes={num_classes}. Check dataset labels and num_output_channels."
            )

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        HCMA training path expects (features, logits), but nnUNet final validation
        predictor expects single logits tensor. Temporarily switch predict_mode to
        True only for full-volume validation, then restore afterwards.
        """
        self._set_predict_mode_recursive(self.network, True)
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self._set_predict_mode_recursive(self.network, False)

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
