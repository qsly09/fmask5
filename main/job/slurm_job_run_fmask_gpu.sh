#!/bin/bash
#SBATCH -J f_gpu
#SBATCH --partition=general-gpu
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-30
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../


DCLOUD=0
DSHADOW=5
SAVEMETA='yes'
DISPLAYFMASK='yes'
DISPLAYIMG='yes'

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset/Sentinel2' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset/Sentinel2MaskGPU' # end names will be provided afterward
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK

MODEL='UPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK

MODEL='UNT'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK


exit



FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset/Sentinel2' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset/Sentinel2MaskGPU_Post' # end names will be provided afterward
MODEL='UNT'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='LPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK

exit

FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat47MaskGPU_Post' # end names will be provided afterward
MODEL='UNT'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='LPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Sentinel2MaskGPU_Post' # end names will be provided afterward
MODEL='UNT'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='LPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK


FOLDER_SRC='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89' # 1350 cores in total
FOLDER_DES='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/Validation/Landsat89MaskGPU_Post' # end names will be provided afterward
MODEL='UNT'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='LPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK
MODEL='UPU'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK


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