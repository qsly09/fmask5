"""
Author: Shi Qiu
Email: shi.qiu@uconn.edu
Date: 2024-05-25
Version: 1.0.0
License: MIT

Description:
This script trains unet models for cloud and shadow detection.

Changelog:
- 1.0.0 (2024-05-24): Initial release.
"""

# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=import-error
# pylint: disable=wrong-import-position
# pylint: disable=no-value-for-parameter
# pylint: disable=syntax-error
import sys
import os
from pathlib import Path
import glob
import click
sys.path.append(str(Path(__file__).parent.parent.joinpath("src")))  # to find necessery libraries based on the absolute path
from fmasklib import Fmask
from utils import exclude_images_by_tile

# define "patch size" and "patch stride", respectively
PATCH_SIZE = 512
PATCH_STRIDE_TRAIN = 488
SAVE_EPOCHS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # save the model every 10 epochs

# below images does not contain cloud shadow layer in L8Biome dataset
IMAGE_LIST_NONSHADOW = ['L895CLOUD', # all the dataset does not have the cloud shadow layer
                        'LC08_L1GT_012055_20130721_20200912_02_T2',
                        'LC08_L1GT_017031_20130606_20200912_02_T2',
                        'LC08_L1GT_029029_20140512_20200911_02_T2',
                        'LC08_L1GT_044116_20131126_20201016_02_T2',
                        'LC08_L1GT_084120_20141105_20201016_02_T2',
                        'LC08_L1GT_102015_20140205_20200912_02_T2',
                        'LC08_L1GT_104062_20140307_20200911_02_T2',
                        'LC08_L1GT_108016_20130620_20200912_02_T2',
                        'LC08_L1GT_117027_20140708_20200911_02_T2',
                        'LC08_L1GT_122042_20140406_20200911_02_T2',
                        'LC08_L1GT_144046_20140907_20200911_02_T2',
                        'LC08_L1GT_151026_20140519_20200911_02_T2',
                        'LC08_L1GT_155008_20140920_20200910_02_T2',
                        'LC08_L1GT_160046_20130803_20200912_02_T2',
                        'LC08_L1GT_166003_20140715_20200911_02_T2',
                        'LC08_L1GT_172019_20131127_20200912_02_T2',
                        'LC08_L1GT_180066_20140818_20200911_02_T2',
                        'LC08_L1GT_192019_20130413_20200912_02_T2',
                        'LC08_L1GT_194022_20130902_20200913_02_T2',
                        'LC08_L1GT_200119_20131201_20201016_02_T2',
                        'LC08_L1GT_227119_20141014_20200910_02_T2',
                        'LC08_L1GT_231059_20140519_20200911_02_T2',
                        'LC08_L1TP_015031_20140814_20200911_02_T1',
                        'LC08_L1TP_018008_20140803_20200911_02_T1',
                        'LC08_L1TP_021007_20140824_20200911_02_T1',
                        'LC08_L1TP_031020_20130811_20200912_02_T1',
                        'LC08_L1TP_034019_20140616_20200911_02_T1',
                        'LC08_L1TP_035019_20140709_20200911_02_T1',
                        'LC08_L1TP_041037_20131223_20200912_02_T1',
                        'LC08_L1TP_042008_20130808_20200912_02_T1',
                        'LC08_L1TP_043012_20140802_20200911_02_T1',
                        'LC08_L1TP_046028_20140620_20200911_02_T1',
                        'LC08_L1TP_050009_20140819_20200911_02_T1',
                        'LC08_L1TP_050017_20140904_20200911_02_T1',
                        'LC08_L1TP_053002_20140605_20200911_02_T1',
                        'LC08_L1TP_063015_20130726_20200912_02_T1',
                        'LC08_L1TP_065018_20130825_20200912_02_T1',
                        'LC08_L1TP_067017_20140725_20200911_02_T1',
                        'LC08_L1TP_076018_20130619_20200912_02_T1',
                        'LC08_L1TP_098076_20140804_20200911_02_T1',
                        'LC08_L1TP_103016_20140417_20200911_02_T1',
                        'LC08_L1TP_107015_20130917_20200912_02_T1',
                        'LC08_L1TP_108018_20140826_20200911_02_T1',
                        'LC08_L1TP_118038_20140901_20200911_02_T1',
                        'LC08_L1TP_124046_20140826_20200911_02_T1',
                        'LC08_L1TP_132035_20130831_20200912_02_T1',
                        'LC08_L1TP_133018_20130705_20200912_02_T1',
                        'LC08_L1TP_133031_20130721_20200912_02_T1',
                        'LC08_L1TP_136030_20140611_20200911_02_T1',
                        'LC08_L1TP_139029_20140515_20200911_02_T1',
                        'LC08_L1TP_146016_20140617_20200911_02_T1',
                        'LC08_L1TP_149012_20130806_20200912_02_T1',
                        'LC08_L1TP_150015_20130813_20200912_02_T1',
                        'LC08_L1TP_157045_20140801_20200911_02_T1',
                        'LC08_L1TP_158017_20130720_20200912_02_T1',
                        'LC08_L1TP_159036_20140220_20200911_02_T1',
                        'LC08_L1TP_166043_20140120_20200912_02_T1',
                        'LC08_L1TP_175043_20130524_20200912_02_T1',
                        'LC08_L1TP_175073_20140204_20200912_02_T1',
                        'LC08_L1TP_197024_20130806_20200912_02_T1',
                        'LC08_L1TP_199040_20140924_20200910_02_T1',
                        'LC08_L1TP_232007_20140814_20200911_02_T1',]

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME",
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder",
)
@click.option(
    "--traindata",
    "-t",
    type=str,
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataCNN512/Landsat8",
    help="The training dataset's directory. If not provided, the results will be saved in the resource directory",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME/UNetCNN512",
    help="The destination directory where the training data will be saved. If not provided, the results will be saved in the resource directory",
)
@click.option("--ci", "-i", type=int, default=1, help="The core's id")
@click.option("--cn", "-n", type=int, default=1, help="The number of cores")
def main(resource, traindata, destination, ci, cn) -> None:
    """
    Generate training data for CNN-based classification.

    Args:
        dataset (str): The training dataset's directory
        resource (str): The path to the resource directory containing the images.
        destination (str): The path to the destination directory where the training data will be saved.
        ci (int): The index of the first task to be processed.
        cn (int): The total number of cores to be used for processing.

    Returns:
        None
    """

    # Create task record object
    tasks = []
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    # exclude the images that do not contain cloud shadow layer
    path_image_list = [i for i in path_image_list if Path(i).stem not in IMAGE_LIST_NONSHADOW]
    for path_image in path_image_list:
        tasks.append({"path_image": path_image})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    for itask, task in enumerate(tasks):
        path_image = task["path_image"]
        name_image = Path(path_image).stem
        print(f"\n> {itask + 1:03d}/{len(tasks):03d} processing {path_image}")

        fmask = Fmask(path_image)
        fmask.init_modules()
        # change the patch size and stride
        fmask.unet_shadow.set_patch_size(PATCH_SIZE)
        fmask.unet_shadow.set_patch_stride_train(PATCH_STRIDE_TRAIN)
        fmask.unet_shadow.set_epoch(max(SAVE_EPOCHS)) # set the maximum epoch
        fmask.unet_shadow.set_train_data_path(traindata) # set the training data path

        # exclude the image that is being processed
        images_excluded,_ = exclude_images_by_tile(exclude = fmask.image.tile) # exclude itself with same tile
        images_excluded = images_excluded + IMAGE_LIST_NONSHADOW # exclude images that does not include shadow reference
        
        # init the training dataset in the unet modelw with an exclude of the image that is being processed
        fmask.unet_shadow.init_train_dataset(patch_index = fmask.get_patch_data_index(fmask.unet_shadow.predictors),
                                             exclude=images_excluded)
        fmask.unet_shadow.train(path = os.path.join(destination, name_image), save_epochs = SAVE_EPOCHS)
        
        del fmask # remove this object from the memory

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
