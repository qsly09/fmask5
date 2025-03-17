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
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89/",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    help="The destination directory. If not provided, the results will be saved in the resource directory",
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89MaskCPU_Post",
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
        print()  # Prints a new line before the progress update
        image_name = Path(path_image).stem
        if "OPER_PRD_" in image_name:
            print(f"{image_name} is not a valid image")
            print(">>> skipping...")
            continue
        if ("LO08_" in image_name) or ("LO09_" in image_name):
            print(f"{image_name} is not a valid image (lack of thermal band)")
            print(">>> skipping...")
            continue
        
        ############### Physical (Fmsak 4.6) ################
        end_key = "PHY"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "physical")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.mask_shadow(postprocess='morphology', min_area=3, potential = "flood")
            # save the mask
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            # save other information
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
            fmask.display_image(bands = ["nir", "red", "green"],
                                title = 'Color composite image',
                                percentiles = [10, 90],
                                path = os.path.join(destination, fmask.image.name + '_NRG.png'))
            fmask.display_image(bands = ["swir1", "nir", "red"],
                                title = 'Color composite image',
                                percentiles = [10, 90],
                                path = os.path.join(destination, fmask.image.name + '_SNR.png'))
            print(f"Finished with {((time.perf_counter() - time_start)/60): 0.2f} mins")

   
        ############### LightGBM ################
        end_key = "GBM"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "lightgbm")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.physical.init_constant_filter() # force to start generate simple masks for snow and water, and the base information for further shadow masking
            fmask.mask_shadow(postprocess='morphology', min_area=3, potential = "flood")
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
            print(f"Finished with {((time.perf_counter() - time_start)/60): 0.2f} mins")
        
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
            fmask.physical.init_constant_filter() # force to start generate simple masks for snow and water, and the base information for further shadow masking
            fmask.mask_shadow(postprocess='none', min_area=0, potential = "flood")
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
            print(f"Finished with {((time.perf_counter() - time_start)/60): 0.2f} mins")
        
        ############### LightGBM Physical LightGBM ################
        end_key = "LPL"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key.upper()}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "interaction", base = "lightgbm", tune = "lightgbm")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.mask_shadow(postprocess='morphology', min_area=3, potential = "flood")
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
            print(f"Finished with {(time.perf_counter() - time_start)/60: 0.2f} mins")
        
        ############### UNet Physical LightGBM ################
        end_key = "UPL"
        if skip and os.path.exists(os.path.join(destination, image_name+ f'_{end_key}.png')):
            print(f"Skip {path_image}")
        else:
            print(f"Processing {path_image} with {end_key}")
            # start the timer
            time_start = time.perf_counter()
            fmask = Fmask(path_image, algorithm = "interaction", base = "unet", tune = "lightgbm")
            fmask.load_image()
            fmask.mask_cloud()
            fmask.mask_shadow(postprocess='unet', min_area=0, potential = "flood")
            fmask.image.destination = destination # force to alter the destination as required
            fmask.save_mask(endname = end_key) # save the mask
            fmask.save_model_metadata(os.path.join(destination, fmask.image.name + f'_{end_key}_meta.csv'), running_time=time.perf_counter() - time_start)
            fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{end_key}.png'))
            print(f"Finished with {(time.perf_counter() - time_start)/60: 0.2f} mins")

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
