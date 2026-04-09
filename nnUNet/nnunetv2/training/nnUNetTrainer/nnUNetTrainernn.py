import os
import torch
from torch import nn
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
