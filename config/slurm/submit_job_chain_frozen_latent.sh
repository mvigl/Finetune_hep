#!/bin/bash

# Default input arguments
EXPROOT="/raven/u/mvigl/public"
OUTDIR="/raven/u/mvigl/public/out"
JOBTIME="12:00:00" # Wall clock limit (max. is 24 hours)
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


config="/raven/u/mvigl/public/Finetune_hep/config/ParT_latent_config.yaml"
checkpoint=""
mess="Frozen_latent"
data="/raven/u/mvigl/public/Finetune_hep/config/train_list.txt"
data_val="/raven/u/mvigl/public/Finetune_hep/config/val_list.txt"
out="/raven/u/mvigl/public/run/Frozen_latent"
Xbb="/raven/u/mvigl/public/run/latent_scores_train/scores/scores.txt"
Xbb_val="/raven/u/mvigl/public/run/latent_scores_val/scores/scores.txt"

#subset=0.0001
#nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_frozen.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$Xbb" "$Xbb_val" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#subset=0.001
#nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_frozen.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$Xbb" "$Xbb_val" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#subset=0.01
#nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_frozen.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$Xbb" "$Xbb_val" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#subset=0.1
#nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_frozen.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$Xbb" "$Xbb_val" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h

subset=1
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_frozen.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$Xbb" "$Xbb_val" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

exit 0
