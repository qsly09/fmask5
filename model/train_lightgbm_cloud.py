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
sys.path.append(str(Path(__file__).parent.parent.joinpath("src")))  # to find necessery libraries based on the absolute path
from fmasklib import Fmask



def train_lightgbm_cloud_model(resource, ci, cn, landsat7 = False) -> None:
    """
    Generate training data for CNN-based classification.

    Args:
        dataset (str): The training dataset's directory
        resource (str): The path to the resource directory containing the images.  L8BIOME for Landsat 7 and 8; S2FMASK
        ci (int): The index of the first task to be processed.
        cn (int): The total number of cores to be used for processing.

    Returns:
        None
    """
    # landsat7 = False # indicate whether we will simulate the landsat7 data using Landsat 8

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

        fmask = Fmask(path_image,algorithm="lightgbm")
        if landsat7:
            fmask.image.spacecraft = "LANDSAT_7"

        fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
        fmask.show_figure = False
        
        fmask.init_pixelbase()
        fmask.pixelbase.load() # load all
        
        fmask.lightgbm_cloud.sample = fmask.pixelbase  # forward the dataset to the random forest model
        fmask.lightgbm_cloud.sample.select()  # select the training samples based on the test setups
        fmask.lightgbm_cloud.train()
        fmask.lightgbm_cloud.save() # save the trained model
        return # just rely on the first image to train the model

# main port to run the fmask by command line
if __name__ == "__main__":
    # just assign the directory of the images, and this script will train the model just based on the first image (do not touch the image data, and just read the pixel training dataset)
    rsc = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME"
    train_lightgbm_cloud_model(rsc, 1, 1, landsat7 = True)
    rsc = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME"
    train_lightgbm_cloud_model(rsc, 1, 1, landsat7 = False)
    rsc = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2FMASK"
    train_lightgbm_cloud_model(rsc, 1, 1, landsat7 = False)
