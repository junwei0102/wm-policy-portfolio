#!/bin/bash
#SBATCH --account=def-gigor_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jwquan/wmpp/logs/%x_%A_%a.out
#SBATCH --job-name=wmpp-p1

# Usage: sbatch --array=1-36 slurm_train_array.sbatch [manifest.csv]
# Reads row $SLURM_ARRAY_TASK_ID of the manifest (skipping the header) and runs
# the unmodified OGBench impls/main.py via the train_policy.py wrapper.

set -euo pipefail

MANIFEST="${1:-/project/6067317/jwquan/wm-policy-portfolio/scripts/p1_manifest.csv}"
RUN_GROUP="${RUN_GROUP:-p1-pilot}"
SAVE_DIR="${SAVE_DIR:-/scratch/jwquan/wmpp/policies/p1}"
VENV=/scratch/jwquan/wmpp/venv
export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls

# wandb offline data goes to node-local tmp and is discarded; train.csv /
# eval.csv in each run dir are the metric record. (A scratch TMPDIR redirect
# previously routed XLA/ptxas temps to scratch and hit the user's quota.)
export WMPP_WANDB_MODE=offline
mkdir -p /scratch/jwquan/wmpp/logs

export MUJOCO_GL=egl

source "$VENV/bin/activate"

# Fail fast if the node's GPU is not visible to JAX: without this, a broken
# CUDA setup silently falls back to CPU (~50x slower) and burns the full
# walltime (observed on fc10512, job 51371597_11).
python -c "import jax; assert any(d.platform == 'gpu' for d in jax.devices())" || {
    echo "[wmpp] PREFLIGHT FAILED: JAX cannot see a GPU on $(hostname); exiting"; exit 1;
}

# tr -d '\r': a CRLF manifest corrupts the last field (e.g. --agent=<path>\r).
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST" | tr -d '\r')
ENV_NAME=$(echo "$ROW" | cut -d, -f2)
ALGO=$(echo "$ROW" | cut -d, -f3)
SEED=$(echo "$ROW" | cut -d, -f4)
FLAGS=$(echo "$ROW" | cut -d, -f5-)

echo "[wmpp] task=$SLURM_ARRAY_TASK_ID env=$ENV_NAME algo=$ALGO seed=$SEED flags=$FLAGS"

# Transient /project mount hiccups can make the agent config unreadable at
# flag-parse time (observed: 3 tasks died with ml_collections _ErrorConfig at
# the same second). Wait up to 2 min for the file instead of dying instantly.
AGENT_FILE="$OGBENCH_IMPLS/agents/${ALGO}.py"
for _ in $(seq 24); do
    [ -r "$AGENT_FILE" ] && break
    echo "[wmpp] waiting for $AGENT_FILE to become readable..."
    sleep 5
done
[ -r "$AGENT_FILE" ] || { echo "[wmpp] PREFLIGHT FAILED: $AGENT_FILE unreadable"; exit 1; }

python /project/6067317/jwquan/wm-policy-portfolio/scripts/train_policy.py \
    --env_name="$ENV_NAME" \
    --seed="$SEED" \
    --eval_episodes=50 \
    --video_episodes=0 \
    --run_group="$RUN_GROUP" \
    --save_dir="$SAVE_DIR" \
    $FLAGS
