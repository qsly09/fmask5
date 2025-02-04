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
from utils import exclude_images_by_tile

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    help="The destination directory. If not provided, the results will be saved in the resource directory",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89Results",
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

    # Create task record object
    tasks = []
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    for path_image in path_image_list:
        tasks.append({"path_image": path_image})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    for task in tasks:
        path_image = task["path_image"]
        image_name = Path(path_image).stem
        # physical rules
        key = "lightgbmphysical"
        if os.path.exists(os.path.join(destination, image_name + f'_{key}.png')):
            print(f"Skip {path_image}")
            continue
        if "OPER_PRD_" in image_name:
            print(f">>> {image_name} is not a valid image")
            print(">>> skipping...")
            continue
        
        if ("LO08_" in image_name) or ("LO09_" in image_name):
            print(f">>> {image_name} is not a valid image (lack of thermal band)")
            print(">>> skipping...")
            continue

        
        time_start = time.perf_counter()
        fmask = Fmask(path_image, algorithm = "interaction")
        fmask.set_trigger("lightgbm")
        fmask.set_tuner("lightgbm")
        fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
        fmask.show_figure = False
        fmask.load_image()

        fmask.physical.init_cloud_probability() #
        
        if fmask.physical.activated:
            fmask.lightgbm_cloud.load_model() # load the trained random forest model for cloud detection
            # fmask.init_pixelbase()
            # # Load the ALL pixel data
            # fmask.pixelbase.load() # load all
            # fmask.rf_cloud.sample.select()  # select the training samples based on the test setups
            # fmask.rf_cloud.train() # train the random forest model for cloud detection
            fmask.mask_cloud()
            # fmask.mask_cloud_post() # post process to reduce commission error from bright surface pixels
            fmask.mask_shadow_geometry(potential = "flood")
        else:
            fmask.mask_cloud_pcp()
            fmask.mask_shadow_pcp()

        fmask.image.destination = destination
        fmask.save_mask(endname = key)
        
        running_time = time.perf_counter() - time_start
        fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{key}_meta.csv'), running_time=running_time)
        
        fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{key}.png'))
        if dataset in ["L8BIOME", "S2ALCD", "S2WHUCDPLUS"]:
            fmask.save_accuracy(dataset,
                                os.path.join(destination, fmask.image.name + f'_{key}_accuracy.csv'),
                                running_time=running_time,
                                shadow=False)

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
