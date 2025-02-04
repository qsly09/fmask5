import sys
import os
from pathlib import Path
import glob
import click


dir = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME/UNetCNN512/"


path_image_list = sorted(glob.glob(os.path.join(dir, "[L|S]*")))

for img in path_image_list:
    
    path_models = glob.glob(os.path.join(img, "unet_nsf_*"))
    for pth in path_models:
        os.system(f"rm -r {pth}")
                            