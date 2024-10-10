#!/bin/bash -l
# Standard output and error:
#SBATCH -o /raven/u/mvigl/public/slurm/job-management/job_%j.out
#SBATCH -e /raven/u/mvigl/public/slurm/job-management/job_%j.err
# Initial working directory:
#SBATCH -D /raven/u/mvigl/public/run
#
# For CPU (uncomment)
# Number of nodes and MPI tasks per node:
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem=100000
#
# For GPU
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mvigl@mpcdf.mpg.de

# Load compiler and MPI modules (must be the same as used in conda env)
module purge
module load anaconda/3/2021.11 # <-> python 3.9.
module load cuda/11.6
module load cudnn/8.8
module load pytorch/gpu-cuda-11.6/2.0.0

# Load conda environment
eval "$(conda shell.bash hook)"
conda env create -f ../gpu_env.yaml
conda activate gpu_env

config="$1"
checkpoint="$2"
mess="$3"
use_hlf="$4"
out="$5"
ishead="$6"
Xbb="$7"
repDim="$8"
data="$9"

# Start run
srun /raven/u/mvigl/public/slurm/job_save_scores.sh "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data"
