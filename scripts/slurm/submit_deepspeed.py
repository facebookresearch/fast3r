import subprocess
import sys

def usage():
    print("Usage: python submit.py --nodes=<NODES> --experiment=<EXPERIMENT>")
    print("  --nodes       Number of nodes to allocate for the job.")
    print("  --experiment  Name of the experiment to be used as the job name.")
    sys.exit(1)

def main():
    # Validate number of arguments
    if len(sys.argv) != 3:
        print("Error: Incorrect number of parameters.")
        usage()

    # Parse parameters
    nodes = None
    experiment = None
    for arg in sys.argv[1:]:
        if arg.startswith('--nodes='):
            nodes = arg.split('=')[1]
        elif arg.startswith('--experiment='):
            experiment = arg.split('=')[1]
        else:
            print(f"Error: Invalid parameter '{arg}'")
            usage()

    # Ensure both parameters are provided
    if not nodes or not experiment:
        print("Error: Missing required parameters.")
        usage()

    # SLURM script as a multi-line string
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={experiment}
#SBATCH --time=7-00:00:00  # 7 days
#SBATCH --mail-user=jianingy@meta.com
#SBATCH --mail-type=BEGIN,END
#SBATCH --account=cortex
#SBATCH --qos=cortex
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive  # Ensures node exclusivity
#SBATCH --signal=SIGUSR1@120  # Send SIGUSR1 120 seconds before job end to allow for checkpointing by Lightning
#SBATCH --output=/opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/research/accel-cortex/dust3r/fast3r/logs/slurm_out/%x-%j.out

echo "Begin setting up env on head node ($HOSTNAME)..."

echo $(env | grep SLURM)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9929
export RDZV_ID=$SLURM_JOBID

export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))  # Critical to ensure dataloaders use all CPUs for torchrun!

. /opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/miniforge3/etc/profile.d/conda.sh
conda activate dust3r

cd /opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/research/accel-cortex/dust3r/fast3r

# Debugging flags (optional)
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "env setup on head node ($HOSTNAME) finished, starting srun..."

# srun --cpu-bind=none \
# torchrun \
#     --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#     src/train.py paths.run_folder_name={experiment}_$SLURM_JOBID trainer.num_nodes={nodes} logger.wandb.name={experiment}_$SLURM_JOBID experiment={experiment}

srun --cpu-bind=none \
    python src/train.py paths.run_folder_name={experiment}_$SLURM_JOBID trainer.num_nodes={nodes} logger.wandb.name={experiment}_$SLURM_JOBID experiment={experiment}

echo "srun finished. Job completed on $(date)"
"""

    # Submit the SLURM job
    process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, text=True)
    process.communicate(slurm_script)


if __name__ == "__main__":
    main()
