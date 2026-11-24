#!/bin/bash
#SBATCH -J cvsi_rbx
#SBATCH -N1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --array=0-5
#SBATCH -o logs/%x_%A_%a.out

# One array task per seed. Best estimator-level setting from rb_harness2 (K=500):
#   stratified-radius rotation, 10 angles per ring, + defensive mixture on 20% of the groups.
#   sbatch run_rbx.sbatch                                   # best:  rotstrat q10 + def0.2
#   sbatch --export=ALL,DEF=0 run_rbx.sbatch                # same sampler without the mixture (comparison arm)
#   sbatch --export=ALL,SEED_LIST="12345 12346 12347 12348 12349 12350 12351 12352 12353 12354 12355 12356 12357 12358 12359" --array=0-14 run_rbx.sbatch   # 15 seeds
# multi_control is left OFF (it gave no gain in the harness).

PY=/home/hice1/hmitra7/scratch/conda_venvs/dem/bin/python
REPO=/home/hice1/hmitra7/scratch/projects/CVSI
cd $REPO
export WANDB_ENTITY=hrishitmitra357-national-institute-of-technology-tiruchi
export WANDB_PROJECT=dem
ENTITY=$WANDB_ENTITY

SEEDS=(${SEED_LIST:-12345 12346 12347 12348 12349 12350})
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
K=${K:-500}; NQ=${NQ:-10}; DEF=${DEF:-0.2}

TAG=rbx_rotstrat_q${NQ}_def${DEF}
NAME=${TAG}_K${K}_seed${SEED}
$PY dem/train.py experiment=gmm_idem trainer=gpu \
    model._target_=dem.models.dem_module_rbx.DEMLitModuleRBX \
    +model.rb_mode=rotstrat +model.rb_nq=$NQ +model.def_alpha=$DEF \
    +model.def_scale=0.6 +model.def_center=0.0 +model.def_min_sigma=0.3 \
    +model.multi_control=false \
    logger=wandb logger.wandb.project=dem +logger.wandb.entity=$ENTITY \
    +logger.wandb.name=$NAME logger.wandb.group=rb_vs_base_K${K} \
    model.num_estimator_mc_samples=$K seed=$SEED \
    hydra.run.dir=$REPO/runs/K${K}/${TAG}/seed${SEED} \
    ++model.version=$((3000 + SLURM_ARRAY_TASK_ID))
