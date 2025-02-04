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


# below images does not contain cloud shadow layer in L8Biome dataset
IMAGE_LIST_NONSHADOW = ['L895CLOUD', # all the dataset does not have the cloud shadow layer
                        'LC08_L1GT_012055_20130721_20200912_02_T2',
                        'LC08_L1GT_017031_20130606_20200912_02_T2',
                        'LC08_L1GT_029029_20140512_20200911_02_T2',
                        'LC08_L1GT_044116_20131126_20201016_02_T2',
                        'LC08_L1GT_084120_20141105_20201016_02_T2',
                        'LC08_L1GT_102015_20140205_20200912_02_T2',
                        'LC08_L1GT_104062_20140307_20200911_02_T2',
                        'LC08_L1GT_108016_20130620_20200912_02_T2',
                        'LC08_L1GT_117027_20140708_20200911_02_T2',
                        'LC08_L1GT_122042_20140406_20200911_02_T2',
                        'LC08_L1GT_144046_20140907_20200911_02_T2',
                        'LC08_L1GT_151026_20140519_20200911_02_T2',
                        'LC08_L1GT_155008_20140920_20200910_02_T2',
                        'LC08_L1GT_160046_20130803_20200912_02_T2',
                        'LC08_L1GT_166003_20140715_20200911_02_T2',
                        'LC08_L1GT_172019_20131127_20200912_02_T2',
                        'LC08_L1GT_180066_20140818_20200911_02_T2',
                        'LC08_L1GT_192019_20130413_20200912_02_T2',
                        'LC08_L1GT_194022_20130902_20200913_02_T2',
                        'LC08_L1GT_200119_20131201_20201016_02_T2',
                        'LC08_L1GT_227119_20141014_20200910_02_T2',
                        'LC08_L1GT_231059_20140519_20200911_02_T2',
                        'LC08_L1TP_015031_20140814_20200911_02_T1',
                        'LC08_L1TP_018008_20140803_20200911_02_T1',
                        'LC08_L1TP_021007_20140824_20200911_02_T1',
                        'LC08_L1TP_031020_20130811_20200912_02_T1',
                        'LC08_L1TP_034019_20140616_20200911_02_T1',
                        'LC08_L1TP_035019_20140709_20200911_02_T1',
                        'LC08_L1TP_041037_20131223_20200912_02_T1',
                        'LC08_L1TP_042008_20130808_20200912_02_T1',
                        'LC08_L1TP_043012_20140802_20200911_02_T1',
                        'LC08_L1TP_046028_20140620_20200911_02_T1',
                        'LC08_L1TP_050009_20140819_20200911_02_T1',
                        'LC08_L1TP_050017_20140904_20200911_02_T1',
                        'LC08_L1TP_053002_20140605_20200911_02_T1',
                        'LC08_L1TP_063015_20130726_20200912_02_T1',
                        'LC08_L1TP_065018_20130825_20200912_02_T1',
                        'LC08_L1TP_067017_20140725_20200911_02_T1',
                        'LC08_L1TP_076018_20130619_20200912_02_T1',
                        'LC08_L1TP_098076_20140804_20200911_02_T1',
                        'LC08_L1TP_103016_20140417_20200911_02_T1',
                        'LC08_L1TP_107015_20130917_20200912_02_T1',
                        'LC08_L1TP_108018_20140826_20200911_02_T1',
                        'LC08_L1TP_118038_20140901_20200911_02_T1',
                        'LC08_L1TP_124046_20140826_20200911_02_T1',
                        'LC08_L1TP_132035_20130831_20200912_02_T1',
                        'LC08_L1TP_133018_20130705_20200912_02_T1',
                        'LC08_L1TP_133031_20130721_20200912_02_T1',
                        'LC08_L1TP_136030_20140611_20200911_02_T1',
                        'LC08_L1TP_139029_20140515_20200911_02_T1',
                        'LC08_L1TP_146016_20140617_20200911_02_T1',
                        'LC08_L1TP_149012_20130806_20200912_02_T1',
                        'LC08_L1TP_150015_20130813_20200912_02_T1',
                        'LC08_L1TP_157045_20140801_20200911_02_T1',
                        'LC08_L1TP_158017_20130720_20200912_02_T1',
                        'LC08_L1TP_159036_20140220_20200911_02_T1',
                        'LC08_L1TP_166043_20140120_20200912_02_T1',
                        'LC08_L1TP_175043_20130524_20200912_02_T1',
                        'LC08_L1TP_175073_20140204_20200912_02_T1',
                        'LC08_L1TP_197024_20130806_20200912_02_T1',
                        'LC08_L1TP_199040_20140924_20200910_02_T1',
                        'LC08_L1TP_232007_20140814_20200911_02_T1',]

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
    default="/scratch/shq19004/ProjectCloudDetectionFmask5/TestOutput/L8BIOME/fmask5/interaction_unet_test_shadow",
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
    # exclude the images that do not contain cloud shadow layer
    path_image_list = [i for i in path_image_list if Path(i).stem not in IMAGE_LIST_NONSHADOW]
    for path_image in path_image_list:
        tasks.append({"path_image": path_image})

    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    for task in tasks:
        path_image = task["path_image"]
        # physical rules
        key = "interaction_unet"
    
        time_start = time.perf_counter()
        fmask = Fmask(path_image, algorithm = "interaction")
        fmask.trigger = "unet"
        fmask.tuner = "unet"
        fmask.optimize()
        fmask.show_figure = False
        fmask.load_data()
        fmask.init_physical_rules()

        fmask.cloud_model_unet_path = os.path.join(
            resource.replace("ReferenceDataset", "TestOutput"),
            "unet_cn_exclude",
            "model",
            fmask.image.name,
            "unet_" + fmask.image.name + "_e{0:03d}.pt".format(fmask.cloud_model_unet_epoch),
        )  # loc to the image being excluded
        
        fmask.shadow_model_unet_path = os.path.join(
            resource.replace("ReferenceDataset", "TestOutput"),
            "unet_sn_exclude",
            "model",
            fmask.image.name,
            "unet_" + fmask.image.name + "_e{0:03d}.pt".format(fmask.shadow_model_unet_epoch),
        )  # loc to the image being excluded

        # use any one of the exsiting unet model to mask the cloud and shadow
        if not os.path.exists(fmask.shadow_model_unet_path):
            fmask.shadow_model_unet_path = os.path.join("/scratch/shq19004/ProjectCloudDetectionFmask5/TestOutput/L8BIOME/unet_sn_exclude/model/LC08_L1GT_001011_20140321_20200911_02_T2", 
                                                        "unet_LC08_L1GT_001011_20140321_20200911_02_T2" + "_e{0:03d}.pt".format(fmask.shadow_model_unet_epoch))  # loc to the image being excluded
        
        if fmask.image.name.startswith("L"):
            fmask.dir_pixel = "/scratch/shq19004/ProjectCloudDetectionFmask5/TrainingDataPixel1/Landsat8"
        else:
            fmask.dir_pixel = "/scratch/shq19004/ProjectCloudDetectionFmask5/TrainingDataPixel1/Sentinel2"

        if fmask.phy.activated:
            fmask.load_cloud_unet()
            fmask.load_shadow_unet()
            fmask.mask_cloud()
            fmask.mask_shadow(potential = "flood")
        else:
            fmask.mask_cloud_pcp()
            fmask.mask_shadow_pcp()
        running_time = time.perf_counter() - time_start

        fmask.image.destination = destination
        # fmask.save_mask(endname = key)
        fmask.save_accuracy(dataset,
                            os.path.join(destination, fmask.image.name + f'_{key}_flood_accuracy.csv'),
                            running_time=running_time,
                            shadow=True)
    
        if fmask.phy.activated:
            fmask.mask_shadow(potential = "unet")
        else:
            fmask.mask_cloud_pcp()
            fmask.mask_shadow_pcp()
        running_time = time.perf_counter() - time_start

        fmask.image.destination = destination
        # fmask.save_mask(endname = key)
        fmask.save_accuracy(dataset,
                            os.path.join(destination, fmask.image.name + f'_{key}_unet_accuracy.csv'),
                            running_time=running_time,
                            shadow=True)
        
        if fmask.phy.activated:
            fmask.mask_shadow()
        else:
            fmask.mask_cloud_pcp()
            fmask.mask_shadow_pcp()
        running_time = time.perf_counter() - time_start

        fmask.image.destination = destination
        # fmask.save_mask(endname = key)
        fmask.save_accuracy(dataset,
                            os.path.join(destination, fmask.image.name + f'_{key}_both_accuracy.csv'),
                            running_time=running_time,
                            shadow=True)
        # fmask.display_fmask(path = os.path.join(destination, fmask.image.name + f'_{key}.png'))

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
