#!/bin/bash
#SBATCH --job-name=flux_transformer
#SBATCH --account=project_2013496
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

CODEDIR=/projappl/project_2013496/fluxformer
WORKDIR=/scratch/project_2013496/$USER/fluxformer

module load python-pytorch/2.10
source /projappl/project_2013496/venvs/vflux-gpu/bin/activate

mkdir -p "$WORKDIR"
cd "$WORKDIR"

srun python3 -u "$CODEDIR/train_flux_transformer_with_cache.py"
