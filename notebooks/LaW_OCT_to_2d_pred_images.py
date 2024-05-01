from pathlib import Path
import SimpleITK as sitk
import cv2
import numpy as np
from matplotlib import colormaps


image_dir = Path('/storage_bizon/naravich/Unlabeled_OCT_by_CADx/')
image_folders = ['CADI', 'CADIII', 'CADIV']

# image_dir = Path('<DESTINATION>')
prediction_dir = Path('/storage_bizon/naravich/Unlabeled_OCT_by_CADx/3d_fullres_predictions')

output_2d_dir = Path('/storage_bizon/naravich/Unlabeled_OCT_by_CADx/2d_fullres_predictions/')
output_2d_dir.mkdir(exist_ok=True)

cmap = colormaps.get_cmap('Set3')
for cnt, prediction_file in enumerate(prediction_dir.glob('*.nii.gz')):
    print(f'Processing {prediction_file}')
    prediction = sitk.ReadImage(str(prediction_file))
    prediction = sitk.GetArrayFromImage(prediction)
    out_folder = output_2d_dir / prediction_file.name.replace('.nii.gz', '')
    out_folder.mkdir(exist_ok=True)
    out_debug_folder = out_folder / 'debug'
    out_debug_folder.mkdir(exist_ok=True)

    found = False
    for folder in image_folders:
        image_file = image_dir / folder / prediction_file.name.replace('.nii.gz', '')
        if image_file.exists():
            found = True
            break
    if not found:
        print(f'Could not find image for {prediction_file}')
        continue
    print(image_file)
    img = cv2.imread
    for i in range(prediction.shape[2]):
        print(f'Saving {i}')
        slice = prediction.get_fdata()[i, :, :].astype(np.uint8).transpose(1, 0)
        colored_slice = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
        for j in range(1, slice.max() + 1):
            colored_slice[slice == j] = [v * 255 for v in cmap(j)[:3]]
        cv2.imwrite(str(out_folder / f'{i:04d}.png'), colored_slice)

        image = cv2.imread(str(image_file / f"{prediction_file.name.replace('.nii.gz', '')}_{i + 1:03d}.png"), cv2.IMREAD_GRAYSCALE)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = image - image.min()
        image = image / image.max()
        image = image * 255
        image = image.astype(np.uint8)
        superposition = cv2.addWeighted(image, 0.6, colored_slice, 0.4, 0)
        cv2.imwrite(str(out_debug_folder / f'{i:04d}.png'), superposition)
        break


    if cnt == 2:
        break