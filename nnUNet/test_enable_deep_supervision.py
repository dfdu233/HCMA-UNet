import sys
import torch
import traceback
from nnunetv2.run.run_training import get_trainer_from_args

try:
    trainer = get_trainer_from_args(1, '3d_fullres', 0, 'nnUNetTrainerSwinUNETRv2', 'nnUNetPlans', False, torch.device('cpu'), '')
    
    print("Trainer initialized.")
    print("enable_deep_supervision:", trainer.enable_deep_supervision)
    
    trainer.initialize()
    print("enable_deep_supervision after initialize:", trainer.enable_deep_supervision)
except Exception as e:
    traceback.print_exc()

