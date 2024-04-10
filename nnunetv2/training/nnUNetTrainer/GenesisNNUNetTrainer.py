from time import time
from typing import Union, List, Tuple
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import wandb
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class GenesisNNUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.frozen = False
        self.freeze_until_epoch = 50

    def on_epoch_start(self):
        super().on_epoch_start()
        if self.num_epochs <= self.freeze_until_epoch and not self.frozen:
            for name, param in self.network.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
            self.frozen = True
            self.print_to_log_file("Frozen network part of the network. Only training the decoder.")
        elif self.num_epochs > self.freeze_until_epoch and self.frozen:
            for param in self.network.parameters():
                param.requires_grad = True
            self.frozen = False
            self.print_to_log_file("Unfrozen network. Training the entire network.")
    

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        network = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

        network_state = torch.load('/storage_bizon/naravich/ModelGenesisNNUNetPretraining/Genesis_OCT_nnUNet.pt')
        new_state_dict = {}
        for k, value in network_state.items():
            key = k
            if key not in network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]

            if 'decoder.seg_layers.4' in key:
                value = network.state_dict()[key]
            new_state_dict[key] = value
        network.load_state_dict(new_state_dict)
        return network


    def initialize_wandb(self, fold: int):
        wandb_project_name = f"{self.__class__.__name__}__{self.plans_manager.dataset_name}__{self.configuration_name}"
        wandb_run_name = f"fold_{fold}"
        wandb.init(
            project=wandb_project_name,
            name=wandb_run_name,
            dir=self.output_folder,
        )