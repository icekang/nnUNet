import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_CalOCT_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
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
    samples = ['101-019', '101-044', '101-045', '106-002', '401-004', '701-013', '704-003', '706-005', '707-003']
    train = ['101-019','101-044','101-045','106-002','401-004','701-013','704-003']
    test = ['706-005', '707-003']
    return train, test

if __name__ == '__main__':
    lawoct_data_dir = Path('/storage_bizon/naravich/3D_Shockwave')

    task_id = 301
    task_name = "Calcium_OCT"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = Path(join(nnUNet_raw, foldername))
    imagestr = Path(join(out_base, "imagesTr"))
    labelstr = Path(join(out_base, "labelsTr"))

    imagestrTest = Path(join(out_base, "imagesTs"))
    labelstrTest = Path(join(out_base, "labelsTs"))
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagestrTest)
    maybe_mkdir_p(labelstrTest)

    train, test = get_splits()
    for case_id in train + test:
        image_path = list((lawoct_data_dir / 'Image').glob(f'{case_id}*'))
        assert len(image_path) != 0, f'Cannot find any image for {case_id}'
        assert len(image_path) == 1, f'Found more than one image for {case_id} {list(image_path)}'
        image_path = image_path[0]

        label_path = list((lawoct_data_dir / 'Calcium').glob(f'{case_id}*'))
        assert len(label_path) != 0, f'Cannot find any label for {case_id}'
        assert len(label_path) == 1, f'Found more than one label for {case_id} {list(label_path)}'
        label_path = label_path[0]
        assert label_path.name == image_path.name, f'Label and image do not match for {case_id}'

        if case_id in train:
            output_image_folder = imagestr
            output_label_folder = labelstr
        else:
            output_image_folder = imagestrTest
            output_label_folder = labelstrTest

        shutil.copy(str(image_path), str(output_image_folder / f'{case_id}_0000.nii.gz'))
        copy_CalOCT_segmentation_and_convert_labels_to_nnUNet(str(label_path),
                                                              str(output_label_folder / f'{case_id}.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'Pre-IVL'},
                          labels={
                              'background': 0,
                              'calcium': 1,
                          },
                          num_training_cases=len(train),
                          file_ending='.nii.gz',
                          regions_class_order=None,
                          license='Private',
                          reference='None',
                          dataset_release='1.0')
