#!/bin/bash
#SBATCH -J rf_phy_tune_update
#SBATCH --partition=priority
#SBATCH --account=zhz18039
#SBATCH --mem-per-cpu=25G
#SBATCH --constraint='epyc128'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-200
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2ALCD' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2ALCD' # end names will be provided afterward
python test_rf_physical_update_rate.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2WHUCDPLUS' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2WHUCDPLUS' # end names will be provided afterward
python test_rf_physical_update_rate.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2FMASK' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2FMASK' # end names will be provided afterward
python test_rf_physical_update_rate.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOMEL7' # end names will be provided afterward
python test_rf_physical_update_rate.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME' # end names will be provided afterward
python test_rf_physical_update_rate.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

echo 'Finished!'
exit

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOMEL7' # end names will be provided afterward
python test_rf_physical_update_rate_disagreement.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME' # end names will be provided afterward
python test_rf_physical_update_rate_disagreement.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2ALCD' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2ALCD' # end names will be provided afterward
python test_rf_physical_update_rate_disagreement.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2WHUCDPLUS' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2WHUCDPLUS' # end names will be provided afterward
python test_rf_physical_update_rate_disagreement.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/S2FMASK' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/S2FMASK' # end names will be provided afterward
python test_rf_physical_update_rate_disagreement.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES



echo 'Finished!'
exit


## backup below, which will not be reached

#!/bin/bash
#SBATCH -J rf
#SBATCH --partition=priority
#SBATCH --account=zhz18039
#SBATCH --mem-per-cpu=50G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-100
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err






#SBATCH --partition=general




#SBATCH --partition=osg
#SBATCH --account=osgusers

