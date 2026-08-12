#!/bin/bash
#SBATCH --job-name=ecoli_samples
#SBATCH --account=project_2013496
#SBATCH --partition=small
#SBATCH --time=71:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

CODEDIR=/projappl/project_2013496/fluxformer
WORKDIR=/scratch/project_2013496/$USER/fluxformer

module load python-data/3.12-31.03
source /projappl/project_2013496/venvs/vflux-cpu/bin/activate

mkdir -p "$WORKDIR/data"
cd "$WORKDIR"

srun python3 -u "$CODEDIR/generate_ecoli_iML1515_AMN_MINN_data.py" \
    --n-samples 1000000 \
    --output-prefix iML1515_AMN_MINN_training_data \
    --model-dir "$CODEDIR/models" \
    --data-dir "$WORKDIR/data"
