#!/bin/bash
#SBATCH --account=def-gigor_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/jwquan/wmpp/logs/%x_%j.out
#SBATCH --job-name=wmpp-wm

# Usage: sbatch [-J name] slurm_train_wm.sbatch <env_name> <horizon> [extra flags...]

set -euo pipefail

ENV_NAME="$1"; HORIZON="$2"; shift 2

VENV=/scratch/jwquan/wmpp/venv
export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls
# wandb mode: trainer defaults to online (compute nodes have internet);
# set WMPP_WANDB_MODE=offline here only if the wandb API is down.
mkdir -p /scratch/jwquan/wmpp/logs

source "$VENV/bin/activate"

python -c "import jax; assert any(d.platform == 'gpu' for d in jax.devices())" || {
    echo "[wmpp] PREFLIGHT FAILED: JAX cannot see a GPU on $(hostname); exiting"; exit 1;
}

python /project/6067317/jwquan/wm-policy-portfolio/scripts/train_world_model.py \
    --env_name="$ENV_NAME" \
    --seed=0 \
    --run_group=wm-pilot \
    --save_dir=/scratch/jwquan/wmpp/wm \
    --train_steps=500000 \
    --wm.horizon="$HORIZON" \
    "$@"
