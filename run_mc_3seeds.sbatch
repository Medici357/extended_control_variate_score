#!/bin/bash
#SBATCH -J cvsi_mc
#SBATCH -N1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x_%A_%a.out

PY=/home/hice1/hmitra7/scratch/conda_venvs/dem/bin/python
cd /home/hice1/hmitra7/scratch/projects/CVSI

SEEDS=(12345 12346 12347)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

PY=/home/hice1/hmitra7/scratch/conda_venvs/dem/bin/python
cd /home/hice1/hmitra7/scratch/projects/CVSI

SEEDS=(12345 12346 12347)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

$PY dem/train.py experiment=gmm_idem_multicontrol trainer=gpu logger=csv \
    'model.cvsi_control_set=[quad,linear]' \
    seed=$SEED ++model.version=$((200 + SLURM_ARRAY_TASK_ID))