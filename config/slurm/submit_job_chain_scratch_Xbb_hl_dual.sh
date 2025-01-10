#!/bin/bash

# Default input arguments
EXPROOT="/raven/u/mvigl/public"
OUTDIR="/raven/u/mvigl/public/out"
JOBTIME="23:50:00" # Wall clock limit (max. is 24 hours)
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


config="/raven/u/mvigl/public/Finetune_hep/config/ParT_alphaXbb_hlf_config.yaml"
checkpoint=""
mess="Scratch_Xbb_hl"
data="/raven/u/mvigl/public/Finetune_hep/config/train_list.txt"
data_val="/raven/u/mvigl/public/Finetune_hep/config/val_list.txt"
out="/raven/u/mvigl/public/run/Scratch_Xbb_hl_dual"
bs=256
se=0

subset=0.0001
Alpha=0.5
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
Alpha=1
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

subset=0.001
Alpha=0.5
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
Alpha=1
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

subset=0.01
Alpha=0.5
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
Alpha=1
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

subset=0.1
Alpha=0.5
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
Alpha=1
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

subset=1
Alpha=0.5
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
Alpha=1
nohup sbatch --job-name="$mess${subset}" --time="${JOBTIME}" single_job_dual.sbatch "$config" "$checkpoint" "$mess" "$data" "$subset" "$data_val" "$out" "$bs" "$Alpha" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h

exit 0
