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
# update_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
update_rates = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
update_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1.0]
update_rates = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 1.0]
update_rates = [0.02, 0.04]
update_rates = [0.01]
update_rates = [0.005, 0.015, 0.020]

update_rates = [0.020]

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
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOMEL7",
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
        for update_rate in update_rates:
            tasks.append({"path_image": path_image, "update_rate":update_rate})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ''
    for task in tasks:
        path_image = task["path_image"]
        update_rate = task["update_rate"]
        print(f"> processing {path_image}")
        print(f"  update_rate: {update_rate}")
       # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "interaction")
        path_csv = os.path.join(
            destination,
            "rf_cloud_physical",
            "update_rate_optimal_seed_percentile",
            fmask0.image.name,
            f"{fmask0.image.name}_{int(1000*update_rate):04d}.csv"
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
        # update the percentile of selecting seed pixels b
        path_csv = os.path.join(
            destination,
            "rf_cloud_physical",
            "update_rate_optimal_seed_percentile",
            fmask.image.name,
            f"{fmask.image.name}_{int(1000*update_rate):04d}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        # Random Forest, set the seed levels for Sentinel-2
        if fmask.image.name[0] == 'S':
            fmask.seed_levels = [75, 75]
        else:
            if LANDSAT7: # for Landsat 7
                # fmask.seed_levels = [0, 0]
                fmask.seed_levels = [25, 25]
            else: # for Landsat 8
                fmask.seed_levels = [25, 25]
        fmask.rf_cloud.tune_update_rate = update_rate # setup the updating rate
        fmask.rf_cloud.tune_append_rate = 0 # setup the appending rate
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
