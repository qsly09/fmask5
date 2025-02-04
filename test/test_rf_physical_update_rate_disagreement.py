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
update_rates = [0.005, 0.015]
update_rates = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.]

# setup the number of samples by 0.005 as interval
update_rates = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

update_rates = [0, 0.0025, 0.005, 0.0075, 0.01]
update_rates = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
update_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

update_rates = [0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0014, 0.0016, 0.0018, 0.0020, 0.0022, 0.0024, 0.0026, 0.0028, 0.0030, 0.0032, 0.0034, 0.0036, 0.0038, 0.0040, 0.0042, 0.0044, 0.0046, 0.0048, 0.0050, 0.0052, 0.0054, 0.0056, 0.0058, 0.0060, 0.0062, 0.0064, 0.0066, 0.0068, 0.0070, 0.0072, 0.0074, 0.0076, 0.0078, 0.0080, 0.0082, 0.0084, 0.0086, 0.0088, 0.0090, 0.0092, 0.0094, 0.0096, 0.0098, 0.01, 0.02, 0.03]
update_rates = [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

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
            "update_rate_optimal_seed_percentile_disagree",
            fmask0.image.name,
            f"{fmask0.image.name}_{int(10000*update_rate):05d}.csv"
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
            fmask.tune_seed = "disagree" # physical
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
        # update the percentile of selecting seed pixels b
        path_csv = os.path.join(
            destination,
            "rf_cloud_physical",
            "update_rate_optimal_seed_percentile_disagree",
            fmask.image.name,
            f"{fmask.image.name}_{int(10000*update_rate):05d}.csv"
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
