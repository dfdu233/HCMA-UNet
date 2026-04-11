import torch
import os
from typing import List, Tuple, Union
from torch import nn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHCMA_SvANet_v2 import (
    nnUNetTrainerHCMA_SvANet_v2,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_SvANet_Lite import (
    HCMA_SvANet_Lite,
)


class nnUNetTrainerHCMA_SvANet_Lite(nnUNetTrainerHCMA_SvANet_v2):
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
        # Lite variant: keep robust trainer logic from v2, but use lighter model defaults.
        self.num_epochs = 100 
        # Stability-first defaults to mitigate non-finite gradients.
        #self.initial_lr = float(os.environ.get("NNUNET_LITE_LR", "6e-5"))
        self.weight_decay = float(os.environ.get("NNUNET_LITE_WD", "1e-2"))
        self.batch_size = 2
        self.num_iterations_per_epoch = 200
        self.max_grad_norm = float(os.environ.get("NNUNET_LITE_MAX_GRAD_NORM", "5.0"))
        self.frloss_warmup_epochs = 2
        # Explicitly disable case_004 gating for Lite experiments.
        self.case004_check_every = 0
        self._case004_reached_threshold = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-5,
        )
        # Parent train_step steps lr_scheduler every iteration, so max_steps must be
        # iteration-based (epochs * iters_per_epoch), otherwise LR collapses to 0
        # within the first epoch.
        max_steps = int(self.num_epochs * self.num_iterations_per_epoch)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, max_steps)
        return optimizer, lr_scheduler

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
        model = HCMA_SvANet_Lite(
            num_input_channels,
            num_output_channels,
            input_size=input_size,
            predict_mode=False,
            # Keep defaults compatible with existing checkpoints when using --c.
            encoder_attn_levels=(0, 2),
            decoder_attn_levels=(0,),
            pool_res=(1, 2),
            use_small_lesion_refine=True,
        )
        return model

    def _maybe_run_case004_gate(self):
        # Keep Lite training free from case-specific periodic validation gates.
        return