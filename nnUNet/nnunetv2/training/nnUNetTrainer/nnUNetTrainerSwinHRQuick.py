from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinHR import nnUNetTrainerSwinHR
import torch


class nnUNetTrainerSwinHRQuick(nnUNetTrainerSwinHR):
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
        self.num_epochs = 4
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 4
