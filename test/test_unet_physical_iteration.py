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
max_iterations = [1, 2, 3, 4, 5]
disagree_rates = [0]

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2WHUCDPLUS",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    help="The destination directory. If not provided, the results will be saved in the resource directory",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2WHUCDPLUS",
)
@click.option("--ci", "-i", type=int, help="The core's id", default=1)
@click.option("--cn", "-n", type=int, help="The number of cores", default=1)
def main(resource, destination, ci, cn) -> None:
    # get the dataset name
    dataset = os.path.basename(resource)

    # Create task record object
    tasks = []
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    for path_image in path_image_list:
        for max_iteration in max_iterations:
            for disagree_rate in disagree_rates:
                tasks.append({"path_image": path_image, "max_iteration":max_iteration, "disagree_rate": disagree_rate})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    image_name_pre = ''
    for task in tasks:
        path_image = task["path_image"]
        max_iteration = task["max_iteration"]
        disagree_rate = task["disagree_rate"]
        print(f"> processing {path_image}")
        print(f"  max_iteration: {max_iteration}")
        print(f"  disagree_rate: {disagree_rate}")

        # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "interaction")
        path_csv = os.path.join(
            destination,
            "unet_cloud_physical",
            "iteration",
            fmask0.image.name,
            f"{fmask0.image.name}_{max_iteration:03d}_{int(100*disagree_rate):03d}.csv"
        )
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue
        
        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            fmask = Fmask(path_image, algorithm = "interaction")
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
        # update this image name
        image_name_pre = os.path.basename(path_image)

        # Load the unet model
        fmask.unet_cloud.load_model()
        # fmask.unet_cloud.tune_epoch = 10 # optimal one for both Landsat and Sentinel-2
        # make it happ  end in one iteration
        fmask.max_iteration = max_iteration
        fmask.disagree_rate = disagree_rate

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
