#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-3


split_index=$(($SLURM_ARRAY_TASK_ID))
export PYTHONPATH=$PYTHONPATH:$(pwd)
if [ $split_index -eq 0 ]
then
    python train_jemp.py --retro 0
elif [ $split_index -eq 1 ]
then
    python train_jemp.py --retro 1
elif [ $split_index -eq 2 ]
then
    python train_t5.py --retro 0
elif [ $split_index -eq 3 ]
then
    python train_t5.py --retro 1
fi



