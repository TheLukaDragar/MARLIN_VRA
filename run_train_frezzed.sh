#!/bin/sh
#SBATCH --job-name=train_MARLIN
#SBATCH --output=train_MARLIN%j.out
#SBATCH --error=train_MARLIN%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 


       
source /ceph/hpc/data/st2207-pgp-users/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate  /ceph/hpc/data/st2207-pgp-users/ldragar/pytorch_env

# export WANDB__SERVICE_WAIT=300
# export WANDB_AGENT_DISABLE_FLAPPING=true #stops wandb agent from flapping 

# wandb_agent=$1
# #script is made to run on 1 node with 1 gpu
# wandb agent $wandb_agent 
module load FFmpeg/4.4.2-GCCcore-11.3.0

srun --nodes=1 --exclusive --gpus=1 --ntasks-per-node=1 --time=2-00:00:00 -p gpu python evaluateVRA.py --config /ceph/hpc/data/st2207-pgp-users/ldragar/MARLIN/config/vra/vra_marlin_ft.yaml --data_path /ceph/hpc/data/st2207-pgp-users/ldragar/ds_marlin/ --num_workers 4 --batch_size 8 --RegressorFeatureOnly

