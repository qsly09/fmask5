"""Test the interaction model below with different setups:
1. test the dilations over the UNet-layer
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

#%% test setups
erosions = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
erosions = [0, 5, 10, 15, 20, 25, 30]
erosions = [0, 10, 20, 30, 40, 50]

NUMBER = 10000
SAMPLING_NONCLOUD = "stratified"
SAMPLING_CLOUD = "stratified"

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    help="The destination directory. If not provided, the results will be saved in the resource directory",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME",
)
@click.option("--ci", "-i", type=int, help="The core's id", default=1)
@click.option("--cn", "-n", type=int, help="The number of cores", default=1)
def main(resource, destination, ci, cn) -> None:
    # get the dataset name
    dataset = os.path.basename(resource)
    # indicate whether we will simulate the landsat7 data using Landsat 8
    if os.path.basename(destination).endswith("L7"):
        LANDSAT7 = True
    else:
        LANDSAT7 = False
    # Create task record object
    tasks = []

    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    for path_image in path_image_list:
        for erosion in erosions:
            tasks.append({"path_image": path_image, "erosion":erosion})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]

    for task in tasks:
        path_image = task["path_image"]
        erosion = task["erosion"]
        print(f"> processing {path_image}")
        print(f"  erosion: {erosion}")
       # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "interaction")
        path_csv = os.path.join(
            destination,
            "water",
            "erosion",
            fmask0.image.name,
            f"{fmask0.image.name}_{int(erosion):04d}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue
        
        # load fmask object once for each image, and if the image is different from the previous one
       
        fmask = Fmask(path_image, algorithm = "interaction")
        if LANDSAT7:
            fmask.image.spacecraft = "LANDSAT_7"
        if LANDSAT7:
            fmask.set_base_machine_learning("lightgbm") # LPL for Landsat 7
        else:
            fmask.set_base_machine_learning("unet") # UPL for Landsat 8 and Sentinel-2
        fmask.set_tune_machine_learning("lightgbm")
        fmask.show_figure = False
        fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
        fmask.init_pixelbase()
        # Load the ALL pixel data, but it will exclude the image's
        # exclude the image that is being processed
        # Load the ALL pixel data, but it will exclude the image's
        images_excluded,_ = exclude_images_by_tile(exclude = fmask.image.tile)
        fmask.pixelbase.set_exclude(images_excluded)
        fmask.pixelbase.load() # load all
        sample_database_all = copy.deepcopy(fmask.pixelbase) # save the sample_all
        
        # loc to the image being excluded
        fmask.unet_cloud.path = os.path.join(
            destination, "UNetCNN512",
            fmask.image.name,
            "unet_ncf_060.pt"
            )  # loc to the image being excluded
        
        fmask.load_image()

        # make it happ  end in one iteration
        fmask.max_iteration = 1
        fmask.tune_seed = 'physical' # make sure the seed is tuned by physical model
        # select the training samples
        fmask.pixelbase = copy.deepcopy(sample_database_all) # give it back
        fmask.lightgbm_cloud.sample = fmask.pixelbase  # forward the dataset to the random forest model
        fmask.lightgbm_cloud.sample.sampling_methods = [SAMPLING_NONCLOUD, SAMPLING_CLOUD] # setup the sampling methods
        fmask.lightgbm_cloud.sample.number = NUMBER # set the number of samples
        fmask.lightgbm_cloud.sample.select()  # select the training samples based on the test setups
        fmask.lightgbm_cloud.train() # train the random forest model for cloud detectiony unet
        
        # Load the unet model
        fmask.unet_cloud.load_model()
        
        # start to process the image by random forest
        time_start = time.perf_counter()
        # fmask.physical.swo_erosion_radius = erosion # set the erosion radius
        fmask.physical.water_erosion_radius = erosion # set the erosion radius
        fmask.physical.init_cloud_probability() #
        
        fmask.mask_cloud_interaction(outcome="classified")
        fmask.create_cloud_object(postprocess="unet", dilation = 100, min_area=3) # 100 pixels dilated unet layer as the potential cloud mask
        
        # fmask.mask_shadow(postprocess='unet', min_area=3, potential = "flood") # unet-based elimination for postpocessing, and also remove the very small cloud objects with less than 3 pixels to reduce peper noises
        
        running_time = time.perf_counter() - time_start
        # set up outputing path

        # save the accuracy of the cloud mask
        fmask.save_accuracy(dataset, path_csv, running_time= running_time) # Save the accuracy of the cloud mask

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
