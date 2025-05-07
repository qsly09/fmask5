'''Test the dilation size for cloud mask in pixels based on the existing masks'''
import os
import glob
import click
import rasterio
from skimage.morphology import binary_dilation
import numpy as np

@click.command()
@click.option(
    "--directory",
    "-d",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89MaskCPU/",
)
@click.option(
    "--key",
    "-k",
    type=str,
    help="The key of matching the mask file, such as UPL for Landsat 8-9 and Sentinel-2; (*) means all",
    default='UPL',
)
@click.option(
    "--label",
    "-l",
    type=int,
    help="The label of the mask to be tested, such as cloud (4), shadow (2), snow (3), and clear (1)",
    default=4,
)
@click.option("--ci", "-i", type=int, help="The core's id", default=1)
@click.option("--cn", "-n", type=int, help="The number of cores", default=1)
def main(directory, key, label, ci, cn) -> None:
    # dilation sizes
    size_dilation = [1, 3, 5, 7, 9, 11, 13, 15]
    # search all the files in the directory
    path_image_list = sorted(glob.glob(os.path.join(directory, f'[L|S]*_{key}.tif')))
    # read the mask file accoridng to ci and cn
    path_image_list = path_image_list[ci-1::cn]
    for path_image in path_image_list:
        with rasterio.open(path_image) as src:
            mask = src.read(1)
            mask_label = mask == label
            for d in size_dilation:
                # making a dilation for the labeling mask
                mask_dilattion = binary_dilation(mask_label, footprint=np.ones((2 * d + 1, 2 * d + 1)), out=None)
                # making a copy of the mask
                mask_save = mask.copy()
                # update the mask with the dilation with the label value
                mask_save[mask_dilattion & (mask != 255)] = label
                path_output = path_image.replace('.tif', f'_d{d:02d}.tif')
                # save the mask
                rasterio.open(path_output, 'w', **src.profile).write(mask_save, 1)
                print(f"Save the mask with dilation size {d} to {path_output}")

if __name__ == "__main__":
    main()
