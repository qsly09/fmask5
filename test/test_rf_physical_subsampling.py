'''This script is to test the physical random forest model's efficiency using sampling approach.'''

"""Test the interaction model below with different setups:
1. The percentile of selecting seed pixels by unet
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
# to test the mininum sample size of triggering physical rules
# subsampling_sizes_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
subsampling_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# only test trigger subsampling when the subsampling size is 1 (means regular sampling is not applied)
subsampling_trigger_mins = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000, 500000]

NUMBER = 10000
SAMPLING_NONCLOUD = "stratified"
SAMPLING_CLOUD = "stratified"

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder",
    default="/scratch/shq19004/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME",
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
        for subsampling_size in subsampling_sizes:
            if subsampling_size > 1:
                subsampling_trigger_min = 0 # when testing subsampling size, the trigger min is 0 means no trigger subsampling
                tasks.append({"path_image": path_image, "subsampling_size":subsampling_size, "subsampling_trigger_min": subsampling_trigger_min,})
            else:
                for subsampling_trigger_min in subsampling_trigger_mins:
                    tasks.append({"path_image": path_image, "subsampling_size":subsampling_size, "subsampling_trigger_min": subsampling_trigger_min,})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ''
    for task in tasks:
        path_image = task["path_image"]
        subsampling_size = task["subsampling_size"]
        subsampling_trigger_min = task["subsampling_trigger_min"]
        print(f"> processing {path_image}")
        print(f"  subsampling_size: {subsampling_size}")
        print(f"  subsampling_trigger_min: {subsampling_trigger_min}")
       # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "interaction")
        path_csv = os.path.join(
            destination,
            "rf_cloud_physical",
            "subsampling",
            fmask0.image.name,
            f"{fmask0.image.name}_{subsampling_trigger_min:09d}_{subsampling_size:02d}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue
        
        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            fmask = Fmask(path_image, algorithm = "interaction")
            if LANDSAT7:
                fmask.image.spacecraft = "LANDSAT_7"
            fmask.set_trigger("randomforest")
            fmask.set_tuner("randomforest")
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
            fmask.load_image()
        # update this image name
        image_name_pre = os.path.basename(path_image)

        # make it happ  end in one iteration
        fmask.max_iteration = 1
        fmask.tune_seed = 'physical' # make sure the seed is tuned by physical model

        # Random Forest, set the seed levels for Sentinel-2
        if fmask.image.name[0] == 'S':
            fmask.seed_levels = [75, 75]
        else:
            if LANDSAT7: # for Landsat 7
                # fmask.seed_levels = [0, 0]
                fmask.seed_levels = [25, 25]
            else: # for Landsat 8
                fmask.seed_levels = [25, 25]
        fmask.rf_cloud.subsampling_size = subsampling_size # set the subsampling size
        fmask.rf_cloud.subsampling_trigger_min = subsampling_trigger_min # set the subsampling trigger min
        # select the training samples
        fmask.pixelbase = copy.deepcopy(sample_database_all) # give it back
        fmask.rf_cloud.sample = fmask.pixelbase  # forward the dataset to the random forest model
        fmask.rf_cloud.sample.sampling_methods = [SAMPLING_NONCLOUD, SAMPLING_CLOUD] # setup the sampling methods
        fmask.rf_cloud.sample.number = NUMBER # set the number of samples
        fmask.rf_cloud.sample.select()  # select the training samples based on the test setups
        fmask.rf_cloud.train() # train the random forest model for cloud detectiony unet

        # start to process the image by random forest
        time_start = time.perf_counter()
        fmask.physical.init_cloud_probability() #
        fmask.mask_cloud_interaction(outcome="classified")
        running_time = time.perf_counter() - time_start
        # set up outputing path

        # save the accuracy of the cloud mask
        fmask.save_accuracy(dataset, path_csv, running_time= running_time) # Save the accuracy of the cloud mask

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
