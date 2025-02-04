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

#%% test setups
tune_epochs = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
tune_epochs = [0, 5, 10, 15, 20]
tune_epochs = [0, 5, 10, 20, 30, 40, 50, 60]
tune_epochs = [0, 5, 10, 15, 20, 25, 30]

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
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    for path_image in path_image_list:
        for tune_epoch in tune_epochs:
            tasks.append({"path_image": path_image, "tune_epoch":tune_epoch})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ''
    for task in tasks:
        path_image = task["path_image"]
        tune_epoch = task["tune_epoch"]
        print(f"> processing {path_image}")
        print(f"  tune_epoch: {tune_epoch}")

        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            fmask = Fmask(path_image, algorithm = "interaction")
            if LANDSAT7:
                fmask.image.spacecraft = "LANDSAT_7"
            fmask.set_trigger("unet")
            fmask.set_tuner("unet")
            fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
            fmask.load_image()
            fmask.show_figure = False
            # loc to the image being excluded
            fmask.unet_cloud.path = os.path.join(
                destination, "UNetCNN512",
                fmask.image.name,
                "unet_ncf_060.pt"
                )  # loc to the image being excluded

        # Load the unet model
        fmask.unet_cloud.load_model()

        # make it happ  end in one iteration
        fmask.max_iteration = 1

        # update the percentile of selecting seed pixels by unet
        fmask.unet_cloud.tune_epoch = tune_epoch

        path_csv = os.path.join(
            destination,
            "unet_cloud_physical",
            "tune_epoch", # "tune_epoch_optimal_seed_percentile",
            fmask.image.name,
            f"{fmask.image.name}_{fmask.unet_cloud.tune_epoch:03d}.csv"
        )
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue
        
        # UNET, set the seed levels for Sentinel-2
        if fmask.image.name[0] == 'S':
            fmask.seed_levels = [0, 0]
            # fmask.seed_levels = [25, 25]
            # fmask.seed_levels = [50, 50]
        else:
            fmask.seed_levels = [0, 0]
        

        # start to process the image by random forest
        time_start = time.perf_counter()
        fmask.physical.init_cloud_probability() #
        fmask.mask_cloud_interaction(outcome="classified")
        running_time = time.perf_counter() - time_start
        # set up outputing path

        # save the accuracy of the cloud mask
        fmask.save_accuracy(dataset, path_csv, running_time= running_time) # Save the accuracy of the cloud mask

        # update this image name
        image_name_pre = os.path.basename(path_image)

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
