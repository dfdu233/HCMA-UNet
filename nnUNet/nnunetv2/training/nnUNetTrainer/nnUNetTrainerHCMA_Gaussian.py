import torch
import os
from typing import List, Tuple, Union
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHCMA import nnUNetTrainerHCMA
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_Gaussian import HCMA_Gaussian


class nnUNetTrainerHCMA_Gaussian(nnUNetTrainerHCMA):
    """HCMA trainer variant with Gaussian small-target supplement."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        exp_name='',
        device: torch.device = torch.device('cuda'),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, exp_name, device)
        self.initial_lr = 1.5e-4
        self.weight_decay = 2e-2
        self.batch_size = 2
        quick_epochs = int(os.environ.get("NNUNET_QUICK_EPOCHS", "100"))
        self.num_epochs = 50 if quick_epochs <= 50 else 100
        # Keep case_004 gating behavior identical to HCMA/HCMA_SvANet_v2.
        self.case004_check_every = 20
        self.case004_dice_threshold = 0.7
        self.case004_key_preferred = "case_004"
        self._case004_reached_threshold = False

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        model = HCMA_Gaussian(
            in_channels=num_input_channels,
            n_classes=num_output_channels,
            input_size=tuple(self.configuration_manager.patch_size),
            predict_mode=False,
        )
        return model
