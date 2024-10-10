#!/bin/bash

# Default input arguments
EXPROOT="/raven/u/mvigl/public"
OUTDIR="/raven/u/mvigl/public/out"
JOBTIME="1-00:00:00" # Wall clock limit (max. is 24 hours)
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
for ((i = 1; i <= NUMJOBS; i++)); do
    JOBNAME="job${i}"
    STRING_ARG="/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml"
    nohup sbatch --job-name="$JOBNAME" --time="${JOBTIME}" single_job_frozen.sbatch "$STRING_ARG" >> "$LOG_FILE" 2>> "$ERR_FILE" &
done

disown -h

exit 0
