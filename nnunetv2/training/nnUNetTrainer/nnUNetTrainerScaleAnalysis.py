import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import random

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    
class nnUNetTrainerScaleAnalysisN(nnUNetTrainer):
    """Scale Analysis Trainer for nnUNet
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), ratio: float = 0.2):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.ratio = ratio

    def do_split(self):
        splits_file = join(self.preprocessed_dataset_folder_base, f"splits_final_{int(self.ratio*100):d}.json")

        if not isfile(splits_file):
            self.print_to_log_file(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            all_keys_sorted = list(np.sort(list(dataset.keys())))
            random.seed(12345)
            splits = []
            for fold_i in range(5):
                random.shuffle(np.arange(len(all_keys_sorted)))
                while True:
                    train_keys = random.choices(all_keys_sorted, k=int(self.ratio * len(all_keys_sorted)))
                    is_duplicate = False
                    for split in splits:
                        is_duplicate = is_duplicate or set(train_keys) == set(split['train'])
                        if is_duplicate:
                            break
                    if not is_duplicate:
                        break
                test_keys = [i for i in all_keys_sorted if i not in train_keys]
                splits.append({})
                splits[-1]['train'] = list(train_keys)
                splits[-1]['val'] = list(test_keys)
            save_json(splits, splits_file)      

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

def nnUNetTrainerScaleAnalysisFactory(ratio: float) -> nnUNetTrainerScaleAnalysisN:
    def constructor(*args, **kwargs):
        return nnUNetTrainerScaleAnalysisN(*args, **kwargs, ratio=ratio)
    return constructor

class nnUNetTrainerScaleAnalysis20(nnUNetTrainerScaleAnalysisN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, ratio=0.2)

class nnUNetTrainerScaleAnalysis40(nnUNetTrainerScaleAnalysisN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, ratio=0.4)

class nnUNetTrainerScaleAnalysis60(nnUNetTrainerScaleAnalysisN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, ratio=0.6)

class nnUNetTrainerScaleAnalysis80(nnUNetTrainerScaleAnalysisN):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, ratio=0.8)