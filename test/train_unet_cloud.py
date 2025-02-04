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
# indicate whether we will simulate the landsat7 data using Landsat 8
LANDSAT7 = True

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
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOMEL7/UNetCNN512",
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
    
    path_image_list = sorted(glob.glob(os.path.join(resource, "[L|S]*")))
    for path_image in path_image_list:
        tasks.append({"path_image": path_image})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    for itask, task in enumerate(tasks):
        path_image = task["path_image"]
        print(f"\n> {itask + 1:03d}/{len(tasks):03d} processing {path_image}")
        
        if not LANDSAT7:
            fmask = Fmask(path_image)
            fmask.init_modules()
            # change the patch size and stride
            fmask.unet_cloud.set_patch_size(PATCH_SIZE)
            fmask.unet_cloud.set_patch_stride_train(PATCH_STRIDE_TRAIN)
            fmask.unet_cloud.set_epoch(max(SAVE_EPOCHS)) # set the maximum epoch
            fmask.unet_cloud.set_train_data_path(traindata) # set the training data path
            # init the training dataset in the unet modelw with an exclude of the image that is being processed
            
            # exclude the image that is being processed
            images_excluded,_ = exclude_images_by_tile(exclude = fmask.image.tile,
                                                    datasets = ["L8BIOME", "L8SPARCS", "L895CLOUD", "S2ALCD", "S2IRIS", "S2WHUCDPLUS", "S2FMASK"],
                                                    directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset") # exclude itself with same tile

            fmask.unet_cloud.init_train_dataset(exclude=images_excluded)
            fmask.unet_cloud.train(path = os.path.join(destination, fmask.image.name), save_epochs = SAVE_EPOCHS)
        else:
            fmask = Fmask(path_image)
            fmask.image.spacecraft = "LANDSAT_7"
            fmask.init_modules()
            # change the patch size and stride
            fmask.unet_cloud.set_patch_size(PATCH_SIZE)
            fmask.unet_cloud.set_patch_stride_train(PATCH_STRIDE_TRAIN)
            fmask.unet_cloud.set_epoch(max(SAVE_EPOCHS)) # set the maximum epoch
            fmask.unet_cloud.set_train_data_path(traindata) # set the training data path
            # init the training dataset in the unet modelw with an exclude of the image that is being processed
            
            # exclude the image that is being processed
            images_excluded,_ = exclude_images_by_tile(exclude = fmask.image.tile,
                                                    datasets = ["L8BIOME", "L8SPARCS", "L895CLOUD", "S2ALCD", "S2IRIS", "S2WHUCDPLUS", "S2FMASK"],
                                                    directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset") # exclude itself with same tile
            # adjust the index of the patch data
            fmask.unet_cloud.init_train_dataset(exclude=images_excluded, patch_index = fmask.get_patch_data_index(fmask.unet_cloud.predictors))
            fmask.unet_cloud.train(path = os.path.join(destination, fmask.image.name), save_epochs = SAVE_EPOCHS)
            
        
        del fmask # remove this object from the memory

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
