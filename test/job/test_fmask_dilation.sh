#!/bin/bash
#SBATCH -J dilation
#SBATCH --partition=priority
#SBATCH --account=zhz18039
#SBATCH --constraint='epyc128' # Target the AMD Epyc node architecture 
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-200
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47MaskCPU'
KEY='LPL'
python test_fmask_dilation.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --directory=$FOLDER_SRC --key=$KEY

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2MaskCPU'
KEY='UPL'
python test_fmask_dilation.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --directory=$FOLDER_SRC --key=$KEY

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89MaskCPU'
KEY='UPL'
python test_fmask_dilation.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --directory=$FOLDER_SRC --key=$KEY




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

#SBATCH --partition=priority
#SBATCH --account=zhz18039

#SBATCH --partition=osg
#SBATCH --account=osgusers