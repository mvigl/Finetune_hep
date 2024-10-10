#!/bin/bash

# Default input arguments
EXPROOT="/raven/u/mvigl/public"
OUTDIR="/raven/u/mvigl/public/out"
JOBTIME="08:30:00" # Wall clock limit (max. is 24 hours)
TLIMIT=23.6
NUMJOBS=1
# Parse input arguments
while getopts ":m:t:e:j:r" o; do
    case "${o}" in
        r)
            EXPROOT=${OPTARG}
            ;;
    o)
        OUTDIR=${OPTARG}
        ;;
    j)
            NUMJOBS=${OPTARG}
            ;;
    esac
done

LOG_FILE="job_script.log"
ERR_FILE="job_script_error.log"

# Submit all jobs
config="/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_config.yaml"
checkpoint="/raven/u/mvigl/public/run/Xbb_task/models/Xbb_task_lr0.001_bs512_subset1.0_epoch_9_Val_loss_0.14788194000720978.pt"
mess="Xbb_scores"
use_hlf=""
save_representaions=""
ishead=""
Xbb=""
repDim=1

data="/raven/u/mvigl/public/Finetune_hep/config/train_list.txt"
out="/raven/u/mvigl/public/run/Xbb_scores_train"
nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

data="/raven/u/mvigl/public/Finetune_hep/config/test_list.txt"
out="/raven/u/mvigl/public/run/Xbb_scores_test"
nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

data="/raven/u/mvigl/public/Finetune_hep/config/val_list.txt"
out="/raven/u/mvigl/public/run/Xbb_scores_val"
nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
exit 0
