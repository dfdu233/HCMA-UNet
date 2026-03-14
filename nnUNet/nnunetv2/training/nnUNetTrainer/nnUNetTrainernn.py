import torch
from torch import nn
from typing import Union, Tuple, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
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
        self.configuration_manager.configuration['batch_size'] = 1
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
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.initial_lr,
            epochs=self.num_epochs,
            pct_start=0.06,
            steps_per_epoch=self.num_iterations_per_epoch,
            anneal_strategy='linear',
        )
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
