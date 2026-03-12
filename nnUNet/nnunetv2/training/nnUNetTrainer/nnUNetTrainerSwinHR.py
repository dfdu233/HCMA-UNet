import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.SwinHR import SwinHR
import math

class nnUNetTrainerSwinHR(nnUNetTrainer):
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
        self.initial_lr = 1e-4
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
        patch_size = tuple(int(s) for s in self.configuration_manager.patch_size)
        model = SwinHR(img_size=patch_size, in_channels=num_input_channels, out_channels=num_output_channels, depths=(1, 1, 1, 1), num_heads=(1, 2, 4, 8), use_checkpoint=True)
        return model
