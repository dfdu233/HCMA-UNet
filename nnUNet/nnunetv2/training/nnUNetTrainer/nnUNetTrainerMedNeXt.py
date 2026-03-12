import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.MedNeXt import MedNeXt
import math

class nnUNetTrainerMedNeXt(nnUNetTrainer):
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
        self.initial_lr = 1e-4
        self.enable_deep_supervision = False
        self.num_epochs = 100
        self.enable_deep_supervision = False  # Custom models mostly don't support deep supervision directly here


    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: list[str] | tuple[str, ...],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        model = MedNeXt(in_channels=num_input_channels, n_channels=32, n_classes=num_output_channels, exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2], kernel_size=3, deep_supervision=False, do_res=True, do_res_up_down=True, block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2], predict_mode=False)
        return model
