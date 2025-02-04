"""
Author: Shi Qiu
Email: shi.qiu@uconn.edu
Date: 2024-11-12
Version: 1.0.0
License: MIT

Description:
This script trains Light GBM for cloud detection.

Changelog:
- 1.0.0 (2024-11-12): Initial release.
"""


# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=import-error
# pylint: disable=wrong-import-position
# pylint: disable=no-value-for-parameter
import sys
import os
import glob
from pathlib import Path
import time
sys.path.append(
    str(Path(__file__).parent.parent.joinpath("src"))
)  # to find the mask based on the absolute path of the package
from fmasklib import Fmask
import click
import copy
from utils import exclude_images_by_tile

dir_im = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME/LC08_L1GT_029029_20140512_20200911_02_T2"
# lightgbm_cloud
fmask = Fmask(dir_im, algorithm = "interaction")
fmask.set_trigger("unet")
fmask.set_tuner("lightgbm")
fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
# fmask.lightgbm_cloud.load_model() # load the trained random forest model for cloud detection
fmask.init_pixelbase()
# # Load the ALL pixel data
fmask.pixelbase.load() # load all
fmask.lightgbm_cloud.sample.select()  # select the training samples based on the test setups
fmask.lightgbm_cloud.train() # train the random forest model for cloud detection
fmask.lightgbm_cloud.save() # save the trained random forest model for cloud detection



dir_im = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME/LC08_L1GT_029029_20140512_20200911_02_T2"
# lightgbm_cloud
fmask = Fmask(dir_im, algorithm = "interaction")
fmask.set_trigger("unet")
fmask.set_tuner("lightgbm")
fmask.image.spacecraft = "LANDSAT_7"
fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
# fmask.lightgbm_cloud.load_model() # load the trained random forest model for cloud detection
fmask.init_pixelbase()
# # Load the ALL pixel data
fmask.pixelbase.load() # load all
fmask.lightgbm_cloud.sample.select()  # select the training samples based on the test setups
fmask.lightgbm_cloud.train() # train the random forest model for cloud detection
fmask.lightgbm_cloud.save() # save the trained random forest model for cloud detection




dir_im = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2FMASK/S2B_MSIL1C_20230101T124119_N0509_R066_T18CVV_20230101T153915.SAFE"
# lightgbm_cloud
fmask = Fmask(dir_im, algorithm = "interaction")
fmask.set_trigger("unet")
fmask.set_tuner("lightgbm")
fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
# fmask.lightgbm_cloud.load_model() # load the trained random forest model for cloud detection
fmask.init_pixelbase()
# # Load the ALL pixel data
fmask.pixelbase.load() # load all
fmask.lightgbm_cloud.sample.select()  # select the training samples based on the test setups
fmask.lightgbm_cloud.train() # train the random forest model for cloud detection
fmask.lightgbm_cloud.save() # save the trained random forest model for cloud detection