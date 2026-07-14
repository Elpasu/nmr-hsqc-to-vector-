#!/bin/bash
#SBATCH --job-name=nmr_v7_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=eval_v7_%j.out
#SBATCH --error=eval_v7_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

cd /home/lpassaglia.iquir/HSQC_to_Vector/V8_Integracion_CH2_numero_senales

python eval_v8.py