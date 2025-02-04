'''test minumnum number of absolute pixels for physical rules'''
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=import-error
# pylint: disable=wrong-import-position
# pylint: disable=no-value-for-parameter
# pylint: disable=f-string-without-interpolation
import sys
import os
import glob
from pathlib import Path
sys.path.append(
    str(Path(__file__).parent.parent.joinpath("src"))
)  # to find the mask based on the absolute path of the package
from fmasklib import Fmask
import click
import time

test_min_clear_number = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory contains mutiple images or a single image folder",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOMEL7",
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
    """main function to start fmask

    Args:
        resource (string): Resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder
        destination (string): Destination directory. If not provided, the results will be saved in the resource directory
        ci (int): Core's id that is consequential number, i.e, 1, 2, 3, ...
        cn (int): Number of cores
    """
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
        for num in test_min_clear_number:
            tasks.append({"path_image": path_image, "min_clear": num})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ""
    for task in tasks:
        path_image = task["path_image"]
        min_clear = task["min_clear"]
        print(f"> processing {path_image}")
        print(f"  mininum number: {min_clear}")

        # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "physical")
        path_csv = os.path.join(
            destination,
            "physical_cloud",
            "min_clear",
            fmask0.image.name,
            f"{fmask0.image.name}_{min_clear:05d}.csv"
        )
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        time_start = time.perf_counter()
        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            # physical rules
            fmask = Fmask(path_image, algorithm = "physical")
            if LANDSAT7:
                fmask.image.spacecraft = "LANDSAT_7"
            fmask.show_figure = False
            fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
            fmask.load_image()

        time_start = time.perf_counter()
        # set the minumum number of clear pixels
        fmask.physical.min_clear = min_clear
        fmask.physical.init_cloud_probability() #
        fmask.mask_cloud_physical()
        running_time = time.perf_counter() - time_start

        fmask.image.destination = destination
        # save the accuracy of the cloud mask
        fmask.save_accuracy(
            dataset, path_csv, running_time=running_time
        )  # Save the accuracy of the cloud mask
        # update this image name
        image_name_pre = os.path.basename(path_image)
# main port to run the fmask by command line
if __name__ == "__main__":
    main()
