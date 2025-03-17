"""
Author: Shi Qiu, Zhe Zhu
Email: shi.qiu@uconn.edu, zhe@uconn.edu
Affiliation: University of Connecticut
Date: 2024-11-12
Version: 5.0.0
License: MIT

Website: https://github.com/gersl/fmask

Description:
This script runs Fmask 5.0 for cloud detection, supporting both Landsat 4-9 and Sentinel-2 data.
The script can process either a single image or an entire directory of images, distributing tasks across multiple cores for efficient processing.

This script runs Fmask 5.0 for cloud detection.
It supports both Landsats 4-9 and Sentinel-2 data.
The script is designed to run on a cluster with multiple cores, with the custumized ci and cn.
The script can be run on a single image or a directory containing multiple images.
The script will divide the tasks into different cores.
The script supports the following algorithms:
- Physical (i.e., Fmask 4.6) (CPU)
- LightGBM (CPU)
- UNet (CPU and GPU)
- LPL: LightGBM-Physical-LightGBM (CPU)
- UPU: UNet-Physical-UNet  (GPU only)
- LPU: LightGBM-Physical-UNet (GPU only)
- UPL: UNet-Physical-LightGBM (CPU and GPU) (default)

Changelog:
- 5.0.0 (2024-11-12): Initial release.
"""

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
import click
import time
sys.path.append(
    str(Path(__file__).parent.parent.joinpath("src"))
)
from fmasklib import Fmask

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset/",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    help="The destination directory. If not provided, the results will be saved in the resource directory",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset/",
)
@click.option("--ci", "-i", type=int, help="The core's id", default=1)
@click.option("--cn", "-n", type=int, help="The number of cores", default=1)
@click.option("--skip", "-s", is_flag = True, help="Skip processing the image when its result exists. (default: False)", default=False)
def main(resource, destination, ci, cn, skip) -> None:
    """main function to start fmask

    Args:
        resource (string): Resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images or a single image folder
        destination (string): Destination directory. If not provided, the results will be saved in the resource directory
        ci (int): Core's id that is consequential number, i.e, 1, 2, 3, ...
        cn (int): Number of cores
    """

    # Create  image list
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    # Select the image folders to process
    path_image_list = [path_image for path_image in path_image_list if os.path.isdir(path_image)]
    # Divide the tasks into different cores
    path_image_list = [path_image_list[i] for i in range(ci - 1, len(path_image_list), cn)] # ci - 1 is the index
    print(f"Core {ci}/{cn}: Processing a total of {len(path_image_list)} images")
   
    # Loop through the images
    for path_image in path_image_list:
        image_name = Path(path_image).stem
        if "OPER_PRD_" in image_name:
            print(f">>> {image_name} is not a valid image")
            print(">>> skipping...")
            continue
        if "LO_" in image_name:
            print(f">>> {image_name} is not a valid image (lack of thermal band)")
            print(">>> skipping...")
            continue

        ############### UNet-Physical-LightGBM ################
        end_key = "UPL"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "interaction", base = "unet", tune = "lightgbm")
            fmask.buffer_cloud = 3 # set the buffer cloud to 3
            fmask.load_image()
            fmask.mask_cloud()
            fmask.create_cloud_object(postprocess = 'unet')
            fmask.mask_shadow()
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
