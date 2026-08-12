#!/bin/bash
#SBATCH --job-name=flux_transformer
#SBATCH --account=project_2013496
#SBATCH --time=69:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

module load python-data
source /scratch/project_2013496/vflux/bin/activate

srun python /projappl/project_2013496/flux_transformer/train_flux_transformer.py
