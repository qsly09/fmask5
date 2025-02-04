"""
Author: Shi Qiu, Zhe Zhu
Email: shi.qiu@uconn.edu, zhe@uconn.edu
Affiliation: University of Connecticut
Date: 2024-11-12
Version: 5.0.0
License: MIT

Website: https://github.com/gersl/fmask

Description:
This script runs Fmask 5.0 for cloud detection in parallel for multiple images using the following models on CPU:
- Physical (i.e., Fmask 4.6) (CPU)
- LightGBM (CPU)
- UNet (CPU and GPU)
- Physical GBM (CPU)
- Physical UNet (GPU only)
- Physical UNet-GBM (CPU and GPU)

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
sys.path.append(
    str(Path(__file__).parent.parent.joinpath("src"))
)
from fmasklib import Fmask
import click
import time

@click.command()
@click.option(
    "--resource",
    "-r",
    type=str,
    help="The resource directory of Landsat/Sentinel-2 data. It supports a directory which contains mutiple images",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    help="The destination directory. If not provided, the results will be saved in the resource directory",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2MaskGPU_Post",
)
@click.option("--ci", "-i", type=int, help="The core's id", default=1)
@click.option("--cn", "-n", type=int, help="The number of cores", default=1)
@click.option("--skip", "-s", type=int, help="skip processing the image when its result exists", default=1)
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
        if ("LO08_" in image_name) or ("LO09_" in image_name):
            print(f">>> {image_name} is not a valid image (lack of thermal band)")
            print(">>> skipping...")
            continue

        ############### UNet ################
        end_key = "UNT"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "unet")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.physical.init_cloud_probability() # force to start from the beginning of the cloud probability, since we do not have the cloud probability from the previous steps
            fmask.create_cloud_object(postprocess = 'none') 
            fmask.mask_shadow()
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
        
        ############### Physical UNet ################
        end_key = "PUU"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "interaction", base = "unet", tune = "unet")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.create_cloud_object(postprocess = 'none') 
            fmask.mask_shadow()
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
    
    
        ############### Physical UNet-GBM ################
        end_key = "PUG"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "interaction", base = "unet", tune = "lightgbm")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.create_cloud_object(postprocess = 'unet') 
            fmask.mask_shadow() # overlap the final cloud mask over the base unet cloud layer to reduce the false positive over bright surfaces 
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
        
        ############### Physical UNet-GBM ################
        end_key = "PGU"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "interaction", base = "lightgbm", tune = "unet")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.create_cloud_object(postprocess = 'none') 
            fmask.mask_shadow()
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))


# main port to run the fmask by command line
if __name__ == "__main__":
    main()
