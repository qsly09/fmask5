#!/bin/bash
#SBATCH -J in_rf
#SBATCH --partition=general
#SBATCH --constraint='epyc128' # Target the AMD Epyc node architecture 
#SBATCH --mem-per-cpu=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-100
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47Results' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2Results' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89Results' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

echo 'Finished!'
exit


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47ResultsPost' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2ResultsPost' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89ResultsPost' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

echo 'Finished!'
exit
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Images/Landsat8' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Masks/Landsat8/randomforest_physical' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Images/Sentinel2' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Masks/Sentinel2/randomforest_physical' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES



FOLDER_SRC='/scratch/shq19004/ProjectCloudDetectionFmask5/ReferenceDataset/S2WHUCDPLUS' # 1350 cores in total
FOLDER_DES='/scratch/shq19004/ProjectCloudDetectionFmask5/Final/S2WHUCDPLUS/randomforest_physical' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/scratch/shq19004/ProjectCloudDetectionFmask5/ReferenceDataset/S2ALCD' # 1350 cores in total
FOLDER_DES='/scratch/shq19004/ProjectCloudDetectionFmask5/Final/S2ALCD/randomforest_physical' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES


FOLDER_SRC='/scratch/shq19004/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # 1350 cores in total
FOLDER_DES='/scratch/shq19004/ProjectCloudDetectionFmask5/Final/L8BIOME/randomforest_physical' # end names will be provided afterward
python final_randomforest_physical.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES


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