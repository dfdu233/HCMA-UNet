import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_SvANet import HCMA_SvANet
import math

class nnUNetTrainerHCMA_SvANet(nnUNetTrainer):
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
        self.num_epochs = 200
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
        model = HCMA_SvANet(num_input_channels, num_output_channels, predict_mode=True)
        return model
