"""
Author: Shi Qiu
Email: shi.qiu@uconn.edu
Date: 2025-03-18
Version: 1.0.0
License: MIT

Description:
This script counts the training data pixels for the training data produced by create_train_data_pixel.py.

Changelog:
- 1.0.0 (2024-05-29): Initial release.
"""

# pylint: disable=no-value-for-parameter
import os
import pandas as pd

def count_training_data(dir_sample, fname="pixelbase.csv"):
    """
    Merge training data from multiple CSV files into a single CSV file.

    Args:
        dir_sample (str): The directory path where the CSV files are located.
        fname (str, optional): The filename of the merged training data CSV file. Defaults to 'training_data.csv'.

    Returns:
        None
    """

    # save the training data
    path_csv = os.path.join(dir_sample, fname)
    
    # load the training data
    training_data = pd.read_csv(path_csv)
    # count the number of training data according to the dataset
    count = training_data["dataset"].value_counts()*1000 # 1/1000 is the ratio of the training data we generated, and see the create_train_data_pixel.py
    print(count)
    print("Total number of training data is: ", count.sum())

# main port to run the fmask by command line
if __name__ == "__main__":
    # type the path of the training data
    count_training_data("/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataPixel1/Landsat8")
    count_training_data("/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataPixel1/Sentinel2")
