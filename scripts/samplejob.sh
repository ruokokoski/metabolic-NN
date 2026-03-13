#!/bin/bash
#SBATCH --job-name=ecoli_samples
#SBATCH --account=project_2013496
#SBATCH --time=10:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load python-data
source /scratch/project_2013496/vflux/bin/activate

srun python /projappl/project_2013496/flux_transformer/generate_ecoli_iML1515_data.py
