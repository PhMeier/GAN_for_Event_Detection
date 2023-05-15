#!/bin/bash
#SBATCH --job-name=nwe
#SBATCH --output=GAN_stockdata_final_noweight.txt
#SBATCH --mail-user=meier@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem 16000
#SBATCH --nodelist=gpu08
#SBATCH --qos=batch

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
cd gan-for-event-detection
cd model
srun python3 -u final_training_no_weight.py #final_training2.py #train2.py # example job step
