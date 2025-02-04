"""Test the unet model below with different setups:
1. The epoch
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

#%% test setups
test_epoch = [1, 5, 10, 20, 40, 60, 80, 100]


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
        for epoch in test_epoch:
            tasks.append({"path_image": path_image, "epoch":epoch})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ''
    for task in tasks:
        path_image = task["path_image"]
        epoch = task["epoch"]
        print(f"> processing {path_image}")
        print(f"  epoch: {epoch}")


        fmask0 = Fmask(path_image, algorithm = "unet")
        # adjust the path of the outcome
        path_csv = os.path.join(
            destination,
            "unet_cloud",
            "epoch",
            fmask0.image.name,
            f"{fmask0.image.name}_{epoch:03d}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            fmask = Fmask(path_image, algorithm = "unet")
            if LANDSAT7:
                fmask.image.spacecraft = "LANDSAT_7"
            fmask.set_trigger("unet")
            fmask.set_tuner("unet")
            fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
            fmask.show_figure = False
            fmask.load_image()

        # update this image name
        image_name_pre = os.path.basename(path_image)

        # adjust the path of the outcome
        path_csv = os.path.join(
            destination,
            "unet_cloud",
            "epoch",
            fmask.image.name,
            f"{fmask.image.name}_{epoch:03d}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        # update the test setups for unet
        fmask.unet_cloud.epoch = epoch
        fmask.unet_cloud.path = os.path.join(
            destination, "UNetCNN512", fmask.image.name,
            f"unet_ncf_{epoch:03d}.pt"
            )  # loc to the image being excluded
        time_start = time.perf_counter()
        fmask.unet_cloud.load_model() # Load the unet model
        fmask.mask_cloud() # Mask cloud by random forest
        running_time = time.perf_counter() - time_start

        # save the accuracy of the cloud mask
        fmask.save_accuracy(dataset, path_csv, running_time= running_time) # Save the accuracy of the cloud mask


# main port to run the fmask by command line
if __name__ == "__main__":
    main()
