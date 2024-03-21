import multiprocessing
import shutil
from multiprocessing import Pool

from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_LaWOCT_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous.
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2]:
            raise RuntimeError('unexpected label')

    shutil.copy(in_file, out_file)


def get_splits():
    """Get the train and test splits for the dataset. This is pre-defined. Please check if the data is still the same.

    Returns:
        Tuple(List[str], List[str]): The train and test splits
    """
    samples = ['AU_MON_00001_Pre_PCI', 'BE_OLV_00008_Pre_PCI', 'BE_OLV_00014_Pre_PCI', 'BE_OLV_00018_Pre_PCI', 'BE_OLV_00020_Pre_PCI', 'BE_OLV_00028_Pre_PCI', 'BE_OLV_00029_Pre_PCI', 'BE_OLV_00031_Pre_PCI', 'BE_OLV_00034_Pre_PCI', 'BE_OLV_00038_Pre_PCI', 'BE_OLV_00048_Pre_PCI', 'BE_OLV_00050_Pre_PCI', 'DK_AHU_00007_Pre_PCI', 'DK_AHU_00015_Pre_PCI', 'DK_AHU_00018_Pre_PCI', 'DK_AHU_00025_Pre_PCI', 'DK_AHU_00027_Pre_PCI', 'JP_KOB_00004_Pre_PCI', 'JP_KOB_00009_Pre_PCI', 'KR_SNC_00008_Pre_PCI']
    train = ['BE_OLV_00034_Pre_PCI', 'BE_OLV_00028_Pre_PCI', 'BE_OLV_00050_Pre_PCI', 'BE_OLV_00018_Pre_PCI', 'JP_KOB_00009_Pre_PCI', 'DK_AHU_00027_Pre_PCI', 'DK_AHU_00015_Pre_PCI', 'BE_OLV_00014_Pre_PCI', 'BE_OLV_00038_Pre_PCI', 'KR_SNC_00008_Pre_PCI', 'BE_OLV_00020_Pre_PCI', 'DK_AHU_00007_Pre_PCI', 'BE_OLV_00031_Pre_PCI', 'BE_OLV_00048_Pre_PCI', 'DK_AHU_00018_Pre_PCI', 'BE_OLV_00029_Pre_PCI']
    test = ['AU_MON_00001_Pre_PCI', 'JP_KOB_00004_Pre_PCI', 'DK_AHU_00025_Pre_PCI', 'BE_OLV_00008_Pre_PCI']
    return train, test
    
if __name__ == '__main__':
    lawoct_data_dir = '/storage_bizon/bizon_imagedata/naravich/3D_OCT_Lumen_Wall'

    task_id = 300
    task_name = "Lumen_and_Wall_OCT"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")

    imagestrTest = join(out_base, "imagesTs")
    labelstrTest = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagestrTest)
    maybe_mkdir_p(labelstrTest)

    train, test = get_splits()
    for case_id in train:
        shutil.copy(join(lawoct_data_dir, f'{case_id}.nii.gz'), join(imagestr, f'{case_id}_0000.nii.gz'))
        copy_LaWOCT_segmentation_and_convert_labels_to_nnUNet(join(lawoct_data_dir, 'labels-lumen-wall-sohee', f'wall_{case_id}_Labls.nii.gz'),
                                                             join(labelstr, case_id + '.nii.gz'))

        train, test = get_splits()
    for case_id in test:
        shutil.copy(join(lawoct_data_dir, f'{case_id}.nii.gz'), join(imagestrTest, f'{case_id}_0000.nii.gz'))
        copy_LaWOCT_segmentation_and_convert_labels_to_nnUNet(join(lawoct_data_dir, 'labels-lumen-wall-sohee', f'wall_{case_id}_Labls.nii.gz'),
                                                             join(labelstrTest, case_id + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'Pre-IVL'},
                          labels={
                              'background': 0,
                              'lumen': (1, ),
                              'wall': (2, ),
                          },
                          num_training_cases=len(train),
                          file_ending='.nii.gz',
                          regions_class_order=None,
                          license='Private',
                          reference='None',
                          dataset_release='1.0')
