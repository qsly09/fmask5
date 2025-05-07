"""Testing Shadow Matching Using the UPL Cloud Detection Model

Test the minimum similarity requirement (using all cloud pixels).
Determine the minimum number of randomly selected cloud pixels needed to match a shadow.
UPL was chosen as it is the most effective cloud detection model for Landsat 8 and Sentinel-2 data.

The same L8Biome dataset was used to assess shadow matching for both Landsat and Sentinel-2.
For Landsat data, the thermal band is utilized to construct a 3D cloud object, helping to narrow the cloud height range.
For Sentinel-2 data, which lacks a thermal band, only a 2D cloud object is used, and the cloud height range remains broader.

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
from multiprocessing import Pool


def process(resource, destination, ci, cn):
    """
    Process function to be called in parallel for cloud and shadow matching.
    This function performs cloud and shadow matching on satellite images using 
    various parameters and configurations. It divides tasks among multiple cores 
    for parallel processing and saves the results to the specified destination.
    Args:
        resource (str): Path to the directory containing the input satellite images.
        destination (str): Path to the directory where the output results will be saved.
        ci (int): Current core index (1-based) for parallel processing.
        cn (int): Total number of cores available for parallel processing.
    Workflow:
        1. Defines various test configurations for similarity, tolerance, penalty weights, 
           thermal inclusion, sampling clouds, and buffer sizes.
        2. Excludes images that do not contain cloud shadow layers based on a predefined list.
        3. Generates tasks for each combination of parameters and divides them among cores.
        4. Processes each task:
            - Loads the image and applies cloud and shadow masking.
            - Configures parameters such as similarity, tolerance, penalty weight, 
              sampling cloud, and thermal inclusion.
            - Saves the results, including accuracy metrics, to the destination directory.
        5. Skips processing if the output CSV file for a task already exists.
    Notes:
        - The function uses the `Fmask` class for cloud and shadow masking.
        - The `Fmask` class is configured with specific algorithms and parameters 
          for processing.
        - The function prints progress and task details during execution.
    Raises:
        Any exceptions raised by the `Fmask` class or file operations.
    Example:
        process("/path/to/resource", "/path/to/destination", 1, 4)
    """

    print(f"Processing core {ci} of {cn}...")
    print(f"Resource: {resource}")
    print(f"Destination: {destination}")

    test_code = 6

    if test_code == 0:
        # test #01
        similarity_tolerances = [0.95] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0] # penalty_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0] # after testing, 0.3 is acceptable for L8Biome dataset; we do not change this parameter. since sometimes matched bad shadow, it the minimum similarity is too low.
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [0]
        buffers = [0]
        connects = [0] # the connect size for the cloud shadow matching proces
        pshadows = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
    elif test_code == 1:
        # test #01
        similarity_tolerances = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0] # penalty_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0] # after testing, 0.3 is acceptable for L8Biome dataset; we do not change this parameter. since sometimes matched bad shadow, it the minimum similarity is too low.
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [0]
        buffers = [0]
        pshadows = [0.15]
        connects = [0] # the connect size for the cloud shadow matching process
    elif test_code == 2: # back to 0.95
        similarity_tolerances = [0.95] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0.9] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # after testing, 0.3 is acceptable for L8Biome dataset
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [0]
        buffers = [0] # the buffer size for the cloud shadow matching process
        connects = [3] # the connect size for the cloud shadow matching process
        pshadows = [0.15]
    elif test_code == 3:
        similarity_tolerances = [0.95] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0] # after testing, 0.3 is acceptable for L8Biome dataset
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [0]
        buffers = [0] # the buffer size for the cloud shadow matching process
        connects = [3] # the connect size for the cloud shadow matching process
        pshadows = [0.15]
    elif test_code == 4:
        similarity_tolerances = [0.95] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0] # after testing, 0.3 is acceptable for L8Biome dataset
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [0]
        connects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        buffers = [0] # the buffer size for the cloud shadow matching process
        pshadows = [0.15]
    if test_code == 5:
        similarity_tolerances = [0.95] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0.9] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0.1] # after testing, 0.3 is acceptable for L8Biome dataset
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [0, 1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000] # the number of cloud pixels to be randomly selected to match shadow
        buffers = [0] # the buffer size for the cloud shadow matching process
        connects = [3] # the connect size for the cloud shadow matching process
        pshadows = [0.15]
    elif test_code == 6:
        similarity_tolerances = [0.95] # the tolerance of the similarity, in order to speed up the process
        penalty_weights = [0.9] # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
        similaritys = [0.1] # after testing, 0.3 is acceptable for L8Biome dataset
        thermal_includes = [0, 1] # yes or no, if the thermal band is included in the cloud shadow matching process
        sampling_clouds = [80000]
        buffers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # the buffer size for the cloud shadow matching process
        connects = [3] # after we get the optimal parameters from the test above, we do the test one more time.
        pshadows = [0.15]
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

    
    
    # get the dataset name
    dataset = os.path.basename(resource)

    # Create task record object
    tasks = []
    path_image_list  = sorted(glob.glob(os.path.join(resource, '[L|S]*')))
    # exclude the images that do not contain cloud shadow layer
    path_image_list = [i for i in path_image_list if Path(i).stem not in IMAGE_LIST_NONSHADOW]
    for path_image in path_image_list:
        for sampling_cloud in sampling_clouds:
            for similarity in similaritys:
                for sim_tol in similarity_tolerances:
                    for penalty_weight in penalty_weights:
                        for thermal_include in thermal_includes:
                            for buf in buffers:
                                for con in connects:
                                    for pshadow in pshadows:
                                        tasks.append({"path_image": path_image, 
                                                      "buffer": buf, "sampling_cloud":sampling_cloud,
                                                      "similarity": similarity, "similarity_tolerance": sim_tol, 
                                                      "penalty_weight": penalty_weight, 
                                                      "thermal_include":thermal_include, 
                                                      "connect":con,
                                                      "pshadow": pshadow})
    # 168 jobs in total for testing the minimum similarity
    # 336 jobs in total for testing how many cloud pixels can be randomly selected to match shadow at mininum
    # Divide the tasks into different cores
    tasks = [tasks[i] for i in range(ci - 1, len(tasks), cn)]
    for task in tasks:
        path_image = task["path_image"]
        similarity = task["similarity"]
        similarity_tolerance = task["similarity_tolerance"]
        penalty_weight = task["penalty_weight"]
        sampling_cloud = task["sampling_cloud"]
        thermal_include = task["thermal_include"]
        buffer = task["buffer"]
        connect = task["connect"]
        pshadow = task["pshadow"]

        print(f"> processing {path_image}")
        print(f"  similarity: {similarity}")
        print(f"  sampling_cloud: {sampling_cloud}")
        print(f"  similarity_tolerance: {similarity_tolerance}")
        print(f"  penalty_weight: {penalty_weight}")
        print(f"  thermal_include: {thermal_include}")
        print(f"  buffer: {buffer}")
        print(f"  connect: {connect}")
        print(f"  pshadow: {pshadow}")

       # check if the csv file exists ahead
        fmask0 = Fmask(path_image, algorithm = "interaction", base = "unet", tune = "lightgbm")
        destination_img = os.path.join(destination, "shadow_match")
        #  "shadow_match_include_thermal",
        #  "shadow_match_exclude_thermal", # also go to match_cloud2shadow funciton to force to set THERMAL_INCLUDED = False
        # end_key = f"{int(1000*similarity):04d}_{sampling_cloud}"
        end_key = f"{thermal_include}_{int(1000*similarity):04d}_{int(1000*similarity_tolerance):04d}_{int(1000*penalty_weight):04d}_{int(1000*pshadow):04d}_{int(buffer)}_{sampling_cloud}"
        end_key = f"{end_key}_{int(connect)}"
        path_csv = os.path.join(
            destination_img,
            fmask0.image.name,
            f"{fmask0.image.name}_{end_key}.csv"
        )  # loc to the image being excluded
        if os.path.exists(path_csv):
            print(f">>> existing {path_csv}")
            print(">>> skipping...")
            continue

        time_start = time.perf_counter()
        
        fmask = Fmask(path_image, algorithm = "interaction", base = "unet", tune = "lightgbm", dcloud = 0, dshadow = buffer, dsnow = 0)
        fmask.physical.similarity = similarity
        fmask.physical.similarity_tolerance = similarity_tolerance
        fmask.physical.penalty_weight = penalty_weight
        fmask.physical.sampling_cloud = sampling_cloud
        fmask.load_image()
        fmask.mask_cloud()
        
        if thermal_include == 1:
            fmask.mask_shadow(postprocess='morphology_unet', min_area=3, potential = "flood", thermal_adjust = True, buffer2connect = connect, threshold = pshadow)
        else:
            fmask.mask_shadow(postprocess='morphology_unet', min_area=3, potential = "flood", thermal_adjust = False, buffer2connect = connect, threshold = pshadow)

        fmask.image.destination = destination_img
        # save the accuracy of the cloud mask
        fmask.save_accuracy(
            dataset, path_csv, running_time=time.perf_counter() - time_start,
            shadow = True
        )  # Save the accuracy of the cloud mask
        # fmask.display_fmask(path = os.path.join(destination_img, fmask.image.name + f'_{end_key}.png'))
        # fmask.display_image(bands = ["nir", "red", "green"],
        #                     title = 'Color composite image',
        #                     percentiles = [10, 90],
        #                     path = os.path.join(destination_img, fmask.image.name + '_NRG.png'))
        # fmask.display_image(bands = ["swir1", "nir", "red"],
        #                     title = 'Color composite image',
        #                     percentiles = [10, 90],
        #                     path = os.path.join(destination_img, fmask.image.name + '_SNR.png'))


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
    default="/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME",
)

@click.option('--ti', type=int, default = 1, required=True, help='SLURM_ARRAY_TASK_ID')
@click.option('--tc', type=int, default = 1, required=True, help='SLURM_ARRAY_TASK_COUNT')
@click.option('--np', type=int, default = 1, required=True, help='SLURM_CPUS_PER_TASK')
def main(resource, destination, ti, tc, np) -> None:
    cn = np * tc # total number of cores
    ts = (ti - 1) * np + 1
    te = ti * np
    ci_list = list(range(ts, te + 1))
    click.echo(f"Total cores: {cn}")
    click.echo(f"Task ID {ti} runs subtasks {ts} to {te} on {np} CPUs")


    # Create a Pool with the number of available CPUs
    with Pool(np) as pool:
        # The `map` function will pass both subtask_id and total_tasks to the process function
        pool.starmap(process, [(resource, destination, ci, cn) for ci in ci_list])
    
    

# main port to run the fmask by command line
if __name__ == "__main__":
    main()
