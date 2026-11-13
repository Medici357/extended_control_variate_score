#!/bin/bash
#SBATCH -J cvsi_rb
#SBATCH -N1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x_%A_%a.out

# One array task per seed (12345, 12346, 12347), each on its own GPU.
#   sbatch --export=ALL,ARM=rb,MODE=radial,RULE=trap,NQ=4,MAXSIG=0.1 run_rb.sbatch   # RB-CVSI arm
#   sbatch --export=ALL,ARM=base run_rb.sbatch                                        # matched baseline: original CVSI
# Knobs (env, defaults): K=500 MODE=coord RULE=strat NQ=10 MAXSIG=null COEF=prop
#
# All runs log to wandb: entity hrishitmitra357-national-institute-of-technology-tiruchi, project dem.
# One-time auth on the login node:  conda activate ~/scratch/conda_venvs/dem && wandb login
# (stores the key in ~/.netrc, which compute nodes share). If compute nodes have no internet:
#   sbatch --export=ALL,ARM=base,WANDB_MODE=offline run_rb.sbatch
#   then on the login node:  wandb sync --sync-all   (from the run dirs under runs/)

PY=/home/hice1/hmitra7/scratch/conda_venvs/dem/bin/python
REPO=/home/hice1/hmitra7/scratch/projects/CVSI
cd $REPO

export WANDB_ENTITY=hrishitmitra357-national-institute-of-technology-tiruchi
export WANDB_PROJECT=dem
ENTITY=$WANDB_ENTITY

SEEDS=(12345 12346 12347)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ARM=${ARM:-rb}; K=${K:-500}; MODE=${MODE:-coord}; RULE=${RULE:-strat}; NQ=${NQ:-10}; MAXSIG=${MAXSIG:-null}; COEF=${COEF:-prop}

nvidia-smi -L

if [ "$ARM" = "base" ]; then
  NAME=cvsi_base_K${K}_seed${SEED}
  $PY dem/train.py experiment=gmm_idem model.use_cvsi=true trainer=gpu \
      logger=wandb logger.wandb.project=dem +logger.wandb.entity=$ENTITY \
      +logger.wandb.name=$NAME logger.wandb.group=rb_vs_base_K${K} \
      model.num_estimator_mc_samples=$K seed=$SEED \
      hydra.run.dir=$REPO/runs/K${K}/base/seed${SEED} \
      ++model.version=$((1000 + SLURM_ARRAY_TASK_ID))
else
  TAG=rb_${MODE}_${RULE}_q${NQ}_max${MAXSIG}_${COEF}
  NAME=${TAG}_K${K}_seed${SEED}
  $PY dem/train.py experiment=gmm_idem_rb trainer=gpu \
      logger=wandb logger.wandb.project=dem +logger.wandb.entity=$ENTITY \
      +logger.wandb.name=$NAME logger.wandb.group=rb_vs_base_K${K} \
      model.num_estimator_mc_samples=$K model.rb_mode=$MODE model.rb_rule=$RULE model.rb_nq=$NQ \
      model.rb_max_sigma=$MAXSIG model.rb_coef_weighting=$COEF seed=$SEED \
      hydra.run.dir=$REPO/runs/K${K}/${TAG}/seed${SEED} \
      ++model.version=$((2000 + SLURM_ARRAY_TASK_ID))
fi
