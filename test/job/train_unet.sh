#!/bin/bash
#SBATCH -J train_unet
#SBATCH --partition=general-gpu
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --array 1-90
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../

#Setting the environment variable CUDA_LAUNCH_BLOCKING=1 can help synchronize the GPU operations and provide a more accurate stack trace
export CUDA_LAUNCH_BLOCKING=1


# 90 images in total for L8BIOME
#  when training the model for Landsat 7, go to the scprit to turn the corresponding flag on
FOLDER_TRAIN_DATA='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataCNN512/Landsat8' # Training patches
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # Reference dataset where we need to process each of the images inside
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOMEL7/UNetCNN512' # Model output
python train_unet_cloud.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --traindata=$FOLDER_TRAIN_DATA --resource=$FOLDER_SRC --destination=$FOLDER_DES

echo 'Finished!'
exit



# 6 images in total for S2FMASK
FOLDER_TRAIN_DATA='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataCNN512/Sentinel2' # Training patches
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2FMASK' # Reference dataset where we need to process each of the images inside
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2FMASK/UNetCNN512' # Model output
python train_unet_cloud.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --traindata=$FOLDER_TRAIN_DATA --resource=$FOLDER_SRC --destination=$FOLDER_DES


# 34 images in total for S2ALCD
FOLDER_TRAIN_DATA='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataCNN512/Sentinel2' # Training patches
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2ALCD' # Reference dataset where we need to process each of the images inside
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2ALCD/UNetCNN512' # Model output
python train_unet_cloud.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --traindata=$FOLDER_TRAIN_DATA --resource=$FOLDER_SRC --destination=$FOLDER_DES


# 36 images in total for S2WHUCDPLUS
FOLDER_TRAIN_DATA='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataCNN512/Sentinel2' # Training patches
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2WHUCDPLUS' # Reference dataset where we need to process each of the images inside
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2WHUCDPLUS/UNetCNN512' # Model output
python train_unet_cloud.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --traindata=$FOLDER_TRAIN_DATA --resource=$FOLDER_SRC --destination=$FOLDER_DES

echo 'Finished!'
exit


# 90 images in total for L8BIOME
FOLDER_TRAIN_DATA='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataCNN512/Landsat8' # Training patches
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # Reference dataset where we need to process each of the images inside
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME/UNetCNN512' # Model output
python train_unet_cloud.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --traindata=$FOLDER_TRAIN_DATA --resource=$FOLDER_SRC --destination=$FOLDER_DES

echo 'Finished!'
exit

## backup below, which will not be reached

#!/bin/bash
#SBATCH -J unet
#SBATCH --partition=priority-gpu
#SBATCH --account=sas18043
#SBATCH --qos=sas18043a100
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=150G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --array 1-9
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


#!/bin/bash
#SBATCH -J unet
#SBATCH --partition=general-gpu
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=150G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --array 1-10
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err





#SBATCH --partition=general




#SBATCH --partition=osg
#SBATCH --account=osgusers

