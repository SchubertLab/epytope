#!/bin/bash -l
#SBATCH -t 0:05:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --array=1
#SBATCH -J pred_bsmall
#SBATCH -o pred_bsmall.out

conda activate tcrconvcuda

export PYTHONPATH="${PYTHONPATH}:/home/icb/anna.chernysheva/tcrconv"

python tcrconv/predictor/pred_tcrconv.py \
--dataset tcrconv/training_data/vdjdb-b-large.csv \
--mode prediction \
--h_cdr31 CDR3B --h_long1 LongB \
--model_file tcrconv/models/statedict_vdjdb-b-large.pt \
--epitope_labels tcrconv/training_data/unique_epitopes_vdjdb-b-large.npy \
--chains B \
--use_LM 1 \
--predfile tcrconv/outputs/preds-b-large.csv --batch_size 256 \
--additional_columns CDR3B Subject
