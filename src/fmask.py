# pylint: disable=line-too-long
import os
from satellite import satellite
import time
from utils import check_image_folder
from pathlib import Path
import randomforest as rf
import pandas as pd
import numpy as np
from rasterio import open as rasopen
import shutil
from skimage.measure import label, regionprops
from fmasklib import Fmask
from rflib import RandomForest
from unetlib import UNet

def fmask(input: str, output = None) -> None:
    """fmask, including cloud, cloud shadow, snow, water, and land

    Args:
        input (string): Resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder
        output (string): Destination directory. If not provided, the results will be saved in the resource directory
    """
    # count the computing time in seconds
    time_start = time.perf_counter()

    # initlize image object, that contains base information regarding this image
    image = satellite(input)
    # init the outputing folder
    if output is not None:
        image.destination = output
    del input, output # clear the variables that will be used anymore
    
    print(f'> processing {image.name}')

    # initialize the configuration
    config = cfg.Config(spacecraft = image.spacecraft)

    # check CUDA and update Configuration with DEVICE
    if config.trigger == "unet" or config.tuner == "unet":
        config.device = unet.check_device()
        print(f'>> running on {config.device}')

    # load data
    print(">> loading data")
    # read saturate qa for visible bands
    # true indicates to update the profile of the image object
    satu = image.read_radiometric_saturation(profile=True)
    # read the datacube according to the predictors provided
    bands, data = image.read_datacube(satu = satu, predictors=config.predictor_full)
    time_elapsed = (time.perf_counter() - time_start) # load all Landsat bands and indcies requires 90 secs
    print (">> in %5.0f secs" % (time_elapsed))
    
    phylab.compute_cloud_probability_layers(
            image, data.copy(), bands, satu)
    # mask clouds
    Fmask = FmaskClass(config) # config fmask
    Fmask.mask_cloud(image, bands, data, satu)
    
    
    # save map
    # initialize the output folder
    Path(image.destination).mkdir(parents=True, exist_ok=True)
        
def batch_fmask(resource: str, destination = None, ci = 1, cn = 1) -> None:
    """batch to start fmask

    Args:
        resource (string): Resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder
        destination (string): Destination directory. If not provided, the results will be saved in the resource directory
        ci (int): Core's id that is consequential number, i.e, 1, 2, 3, ...
        cn (int): Number of cores
    """

    # check image folder(s) in the directory inputted
    path_image_list = check_image_folder(resource)
    
    # Process each image found
    for i in range(ci - 1, len(path_image_list), cn):
        if destination is None:
            
            fmask(path_image_list[i])
        else:
            fmask(path_image_list[i], path_image_list[i])
            