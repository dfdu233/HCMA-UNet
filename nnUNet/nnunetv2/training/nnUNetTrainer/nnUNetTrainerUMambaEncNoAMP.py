import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class nnUNetTrainerUMambaEncNoAMP(nnUNetTrainer):
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
        self.max_grad_norm = 8.0
        self._last_finite_loss = 1.0
        self._nonfinite_batches = 0

    @staticmethod
    def _is_finite_tensor(x) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).all())
        if isinstance(x, (list, tuple)):
            return all(nnUNetTrainerUMambaEncNoAMP._is_finite_tensor(i) for i in x)
        return True

    def _grads_are_finite(self) -> bool:
        for p in self.network.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        if len(self.configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(self.plans_manager, self.dataset_json, self.configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(self.configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(self.plans_manager, self.dataset_json, self.configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        return model

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._nonfinite_batches = 0


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        if not self._is_finite_tensor(output):
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(
                f"WARNING: non-finite logits encountered, skipping batch (count={self._nonfinite_batches})"
            )
            return {'loss': float(self._last_finite_loss)}

        l = self.loss(output, target)
        if not torch.isfinite(l):
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(
                f"WARNING: non-finite loss encountered, skipping batch (count={self._nonfinite_batches})"
            )
            return {'loss': float(self._last_finite_loss)}

        l.backward()
        if not self._grads_are_finite():
            self.optimizer.zero_grad(set_to_none=True)
            self._nonfinite_batches += 1
            self.print_to_log_file(
                f"WARNING: non-finite gradients detected, skipping optimizer step (count={self._nonfinite_batches})"
            )
            return {'loss': float(self._last_finite_loss)}

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._last_finite_loss = float(l.detach().cpu().item())
        return {'loss': self._last_finite_loss}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        if not self._is_finite_tensor(output):
            self.print_to_log_file("WARNING: non-finite logits encountered during validation step")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        del data
        l = self.loss(output, target)
        if not torch.isfinite(l):
            self.print_to_log_file("WARNING: non-finite validation loss encountered")
            l = torch.as_tensor(float(self._last_finite_loss), device=output.device)

        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(target, (list, tuple)):
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