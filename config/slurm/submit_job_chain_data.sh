#!/bin/bash

# Default input arguments
EXPROOT="/raven/u/mvigl/public"
OUTDIR="/raven/u/mvigl/public/out"
JOBTIME="05:59:00" # Wall clock limit (max. is 24 hours)
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


config="/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml"
checkpoint=""
mess="Scratch_Xbb_hl"
data="/raven/u/mvigl/public/Finetune_hep/config/train_list.txt"
data_val="/raven/u/mvigl/public/Finetune_hep/config/val_list.txt"
subset=1
nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" >> "$LOG_FILE" 2>> "$ERR_FILE" &

disown -h

exit 0
