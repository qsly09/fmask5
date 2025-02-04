#!/bin/bash
#SBATCH -J phy_shadow
#SBATCH --partition=general
#SBATCH --mem-per-cpu=30G
#SBATCH --constraint='epyc128'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-140
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../

# 240 tasks in total for testing the similarity
# 140 tasks for testing number of sampling pixels
# one for having thermal band, and one for not having thermal band
FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset/L8BIOME' #
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Test/L8BIOME' # end names will be provided afterward
python test_physical_shadow_match.py --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --resource=$FOLDER_SRC --destination=$FOLDER_DES

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

