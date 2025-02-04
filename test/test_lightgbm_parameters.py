"""Test the random forest model below with different setups:
1. The number of training pixels
2. The sampling strategy of cloud pixels and non-cloud pixels
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

# %% test setups
test_training_number = [10000] # which was found at the default parameters of this model, that is 31 for the number of leaves and 20 for the minimum data in leaf
test_sampling_cloud = ["stratified"]
test_sampling_noncloud = ["stratified"]

# test the three key parameters of LightGBM
# see https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# self.num_leaves = num_leaves # num_leaves = 31 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
# self.min_data_in_leaf = min_data_in_leaf # min_data_in_leaf = 20 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
# self.max_depth = max_depth # max_depth = -1 (no limit) https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

# setup for testing num_leaves and min_data_in_leaf under the same max_depth (no limit) and ntrees (100)
test_ntrees = [100] # that we do not change the number of trees
test_num_leaves = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100] # default value is 31
# test_min_data_in_leaf = [10, 20, 30, 40, 50, 60, 70, 80, 100] # default value is 20
test_min_data_in_leaf = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] # default value is 20
test_max_depth = [-1] # -1 means no limit, None in the code

test_num_leaves = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800] # default value is 31
test_min_data_in_leaf = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800] # default value is 20

test_num_leaves       = [20, 40, 60, 80, 100, 120, 140, 160] # default value is 31
test_min_data_in_leaf = [100, 200, 300, 400, 500, 600, 700, 800] # default value is 20

test_num_leaves       = [5, 10, 15, 20, 25, 30, 35, 40] # default value is 31
test_min_data_in_leaf = [100, 200, 300, 400, 500, 600, 700, 800] # default value is 20


# only keep the same sampling strategy for cloud and non-cloud pixels in the following code
CONTROL_SAME_STRATEGY = True


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
    path_image_list = sorted(glob.glob(os.path.join(resource, "[L|S]*")))

    for path_image in path_image_list:
        for num in test_training_number:
            for sampling_cloud in test_sampling_cloud:
                for sampling_noncloud in test_sampling_noncloud:
                    for ntrees in test_ntrees:
                        for max_depth in test_max_depth:
                            for num_leave in test_num_leaves:
                                for min_data_in_leaf in test_min_data_in_leaf:
                                    # only keep the same sampling strategy for cloud and non-cloud pixels
                                    # random vs. stratified
                                    if CONTROL_SAME_STRATEGY & (sampling_cloud != sampling_noncloud):
                                        continue
                                    # append to tasks
                                    tasks.append(
                                        {
                                            "path_image": path_image,
                                            "number": num,
                                            "sampling_cloud": sampling_cloud,
                                            "sampling_noncloud": sampling_noncloud,
                                            "ntrees": ntrees,
                                            "max_depth": max_depth,
                                            "num_leaves": num_leave,
                                            "min_data_in_leaf": min_data_in_leaf,
                                        }
                                    )
                   
    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ""
    for task in tasks:
        path_image = task["path_image"]
        number = task["number"]
        sampling_cloud = task["sampling_cloud"]
        sampling_noncloud = task["sampling_noncloud"]
        ntrees = task["ntrees"]
        max_depth = task["max_depth"]
        num_leaves = task["num_leaves"]
        min_data_in_leaf = task["min_data_in_leaf"]
        
        print(f"> processing {path_image}")
        print(f"  training number: {number}")
        print(f"  sampling cloud: {sampling_cloud}")
        print(f"  sampling noncloud: {sampling_noncloud}")
        print(f"  ntrees: {ntrees}")
        print(f"  max_depth: {max_depth}")
        print(f"  num_leaves: {num_leaves}")
        print(f"  min_data_in_leaf: {min_data_in_leaf}")

        # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "lightgbm")
        # define the destination of the output
        image_dest = os.path.join(
            destination,
            f"lightgbm_cloud_{sampling_cloud}_{sampling_noncloud}",
            "num_leaves_min_data_in_leaf_max_depth",
            fmask0.image.name
        )
        image_filename = f"{fmask0.image.name}_{number:05d}_{ntrees:05d}_{max_depth:05d}_{num_leaves:05d}_{min_data_in_leaf:05d}"
        path_csv = os.path.join(image_dest,
            f"{image_filename}.csv"
        )

        # skip the existing cloud mask
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            # % Create an instance of Fmask with the image's directory
            fmask = Fmask(path_image, algorithm="lightgbm")
            # force to set up the spacecraft as LANDSAT_7 for testing
            if LANDSAT7:
                fmask.image.spacecraft = "LANDSAT_7"
            fmask.show_figure = False
            # Define the directory of the sample training pixels according to the spacecraft
            fmask.init_modules()  # Initialize the random forest model
            # Initialize the database of the training pixels
            fmask.init_pixelbase()
            # Load the ALL pixel data, but it will exclude the image's
            images_excluded,_ = exclude_images_by_tile(exclude = fmask.image.tile)
            fmask.pixelbase.set_exclude(images_excluded)
            fmask.pixelbase.load()
            # Restoring the sample database
            sample_database_all = copy.deepcopy(fmask.pixelbase)
            # Load data that will be used in the following steps
            fmask.load_image()
        else:
            print(">>> skipping initilizing Fmask object for same image")

        # update the destination of the cloud mask
        fmask.image.destination = image_dest
        path_csv = os.path.join(
            fmask.image.destination, 
            f"{image_filename}.csv"
        )
        # skip the existing cloud mask
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        fmask.pixelbase = copy.deepcopy(sample_database_all) # give it back
        fmask.lightgbm_cloud.sample = fmask.pixelbase  # forward the dataset to the random forest model
        fmask.lightgbm_cloud.sample.sampling_methods = [sampling_noncloud, sampling_cloud]
        fmask.lightgbm_cloud.sample.number = number
        fmask.lightgbm_cloud.ntrees = ntrees
        fmask.lightgbm_cloud.max_depth = max_depth
        fmask.lightgbm_cloud.num_leaves = num_leaves
        fmask.lightgbm_cloud.min_data_in_leaf = min_data_in_leaf
        fmask.lightgbm_cloud.sample.select()  # select the training samples based on the test setups
        fmask.lightgbm_cloud.train()

        time_start = time.perf_counter()
        fmask.mask_cloud()  # Mask cloud by random forest
        running_time = time.perf_counter() - time_start

        # save the accuracy of the cloud mask
        fmask.save_accuracy(
            dataset, path_csv, running_time=running_time
        )  # Save the accuracy of the cloud mask

        # save the importance of the random forest model
        path_csv = os.path.join(
            fmask.image.destination,
            f"{image_filename}_predictor_importance.csv"
        )
        fmask.lightgbm_cloud.save_importance(
            path=path_csv
        )  # Save the importance of the random forest model

        # update this image name
        image_name_pre = os.path.basename(path_image)


# main port to run the fmask by command line
if __name__ == "__main__":
    main()
