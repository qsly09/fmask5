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
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME",
)


@click.option("--ci", "-i", type=int, help="The core's id", default=1)
@click.option("--cn", "-n", type=int, help="The number of cores", default=1)
def main(resource, destination, ci, cn) -> None:
    # get the dataset name
    dataset = os.path.basename(resource)

    # Create task record object
    tasks = []
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    # exclude the images that do not contain cloud shadow layer
    path_image_list = [i for i in path_image_list if Path(i).stem not in IMAGE_LIST_NONSHADOW]
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

        # load fmask object once for each image, and if the image is different from the previous one
        if image_name_pre != os.path.basename(path_image):
            fmask = Fmask(path_image, algorithm = "physical")
            fmask.init_modules() # initilize the modules, such as physical, rf_cloud, unet_cloud, etc.
            fmask.show_figure = False
            fmask.load_image()

        # update this image name
        image_name_pre = os.path.basename(path_image)

        # adjust the path of the outcome
        path_csv = os.path.join(
            destination,
            "unet_shadow",
            "epoch",
            fmask.image.name,
            f"{fmask.image.name}_{epoch:03d}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            # continue

        # update the test setups for unet
        fmask.unet_shadow.epoch = epoch
        fmask.unet_shadow.path = os.path.join(
            destination, "UNetCNN512", fmask.image.name,
            f"unet_nsf_{epoch:03d}.pt"
            )  # loc to the image being excluded
        time_start = time.perf_counter()
        fmask.unet_shadow.load_model() # Load the unet model
        fmask.buffer_shadow = 0
        fmask.shadow = fmask.mask_shadow_unet()
        running_time = time.perf_counter() - time_start

        # save the accuracy of the cloud mask
        fmask.save_accuracy(dataset, path_csv, running_time= running_time, shadow = True) # Save the accuracy of the cloud mask


# main port to run the fmask by command line
if __name__ == "__main__":
    main()
