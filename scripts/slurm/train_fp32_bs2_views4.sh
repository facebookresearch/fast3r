#!/bin/bash
#SBATCH --job-name=fp32_bs2_views4
#SBATCH --time=14-00:00:00  # 14 days
#SBATCH --mail-user=jianingy@meta.com
#SBATCH --mail-type=BEGIN,END
#SBATCH --account=cortex
#SBATCH --qos=cortex_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@120  # Send SIGUSR1 120 seconds before job end to allow for checkpointing by Lightning
#SBATCH --output=/opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/research/accel-cortex/dust3r/fast3r/logs/slurm_out/%x-%j.out

echo "Begin setting up env on head node ($HOSTNAME)..."

echo $(env | grep SLURM)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9929
export RDZV_ID=$SLURM_JOBID

export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))    # this is critical to ensure dataloaders uses all CPUs for torchrun!

. /opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/miniforge3/etc/profile.d/conda.sh
conda activate dust3r

cd /opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/research/accel-cortex/dust3r/fast3r

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "env setup on head node ($HOSTNAME) finished, starting srun..."

# --cpu-bind=none is critical to ensure that the dataloaders can use all CPUs
srun --cpu-bind=none --jobid $SLURM_JOBID /bin/bash -c ' \   # very important to use single quote here so that the variables are not expanded
echo MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT, SLURM_PROCID: $SLURM_PROCID && \
echo local hostname: $(hostname) && \
torchrun \
    --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train.py experiment=fp32_bs2_views4
'

echo "srun finished. Job completed on $(date)"
