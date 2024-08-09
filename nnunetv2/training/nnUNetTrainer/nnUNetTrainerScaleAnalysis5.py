import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import wandb
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainerScaleAnalysis5(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.print_to_log_file(f'Using {default_num_processes} processes for validation.')
        self.print_to_log_file(f'Using {get_allowed_n_proc_DA()} processes for data augmentation.')

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final_5.json")

            if not isfile(splits_file):
                raise ValueError("splits_final_5.json does not exist. You need to create it.")

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                raise ValueError("You requested fold %d for training but splits contain only %d folds.")

            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def initialize_wandb(self, fold: int):
        wandb_project_name = f"{self.__class__.__name__}__{self.plans_manager.dataset_name}__{self.configuration_name}"
        wandb_run_name = f"fold_{fold}"
        wandb.init(
            project=wandb_project_name,
            name=wandb_run_name,
            dir=self.output_folder,
        )