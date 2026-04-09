import torch
import numpy as np
from torch import nn
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHCMA import nnUNetTrainerHCMA

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class nnUNetTrainerHCMA_SvANet_Lite(nnUNetTrainer):
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
    def build_network_architecture(self):
        base_model = super().build_network_architecture()
        model = HCMA_SvANet_Lite(base_model)
        return model