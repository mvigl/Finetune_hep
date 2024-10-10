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
mess="SB_scores"
use_hlf=""
save_representaions=""
ishead=""
Xbb="/raven/u/mvigl/public/run/Xbb_scores_test/scores/scores.txt"
repDim=1
data="/raven/u/mvigl/public/Finetune_hep/config/test_list.txt"

##Scratch_Xbb_hl
#config="/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml"
#checkpoint="/raven/u/mvigl/public/run/Scratch_Xbb_hl/models/Scratch_Xbb_hl_lr0.001_bs256_subset0.0001.pt"
#out="/raven/u/mvigl/public/run/Scratch_Xbb_hl/scores/0_0001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_Xbb_hl/models/Scratch_Xbb_hl_lr0.001_bs256_subset0.001.pt"
#out="/raven/u/mvigl/public/run/Scratch_Xbb_hl/scores/0_001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_Xbb_hl/models/Scratch_Xbb_hl_lr0.001_bs256_subset0.01.pt"
#out="/raven/u/mvigl/public/run/Scratch_Xbb_hl/scores/0_01/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_Xbb_hl/models/Scratch_Xbb_hl_lr0.001_bs256_subset0.1.pt"
#out="/raven/u/mvigl/public/run/Scratch_Xbb_hl/scores/0_1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_Xbb_hl/models/Scratch_Xbb_hl_lr0.001_bs256_subset1.0.pt"
#out="/raven/u/mvigl/public/run/Scratch_Xbb_hl/scores/1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#
##Finetune_Xbb_hl
#config="/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml"
#checkpoint="/raven/u/mvigl/public/run/Finetune_Xbb_hl/models/Finetune_Xbb_hl_lr0.001_bs256_subset0.0001.pt"
#out="/raven/u/mvigl/public/run/Finetune_Xbb_hl/scores/0_0001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_Xbb_hl/models/Finetune_Xbb_hl_lr0.001_bs256_subset0.001.pt"
#out="/raven/u/mvigl/public/run/Finetune_Xbb_hl/scores/0_001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_Xbb_hl/models/Finetune_Xbb_hl_lr0.001_bs256_subset0.01.pt"
#out="/raven/u/mvigl/public/run/Finetune_Xbb_hl/scores/0_01/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_Xbb_hl/models/Finetune_Xbb_hl_lr0.001_bs256_subset0.1.pt"
#out="/raven/u/mvigl/public/run/Finetune_Xbb_hl/scores/0_1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_Xbb_hl/models/Finetune_Xbb_hl_lr0.001_bs256_subset1.0.pt"
#out="/raven/u/mvigl/public/run/Finetune_Xbb_hl/scores/1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#
##Scratch_latent_hl
#config="/raven/u/mvigl/public/Finetune_hep/config/ParT_latent_hlf_config.yaml"
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent_hl/models/Scratch_latent_hl_lr0.001_bs256_subset0.0001.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent_hl/scores/0_0001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent_hl/models/Scratch_latent_hl_lr0.001_bs256_subset0.001.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent_hl/scores/0_001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent_hl/models/Scratch_latent_hl_lr0.001_bs256_subset0.01.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent_hl/scores/0_01/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent_hl/models/Scratch_latent_hl_lr0.001_bs256_subset0.1.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent_hl/scores/0_1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent_hl/models/Scratch_latent_hl_lr0.001_bs256_subset1.0.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent_hl/scores/1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#
##Finetune_latent_hl
#config="/raven/u/mvigl/public/Finetune_hep/config/ParT_latent_hlf_config.yaml"
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent_hl/models/Finetune_latent_hl_lr0.001_bs256_subset0.0001.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent_hl/scores/0_0001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent_hl/models/Finetune_latent_hl_lr0.001_bs256_subset0.001.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent_hl/scores/0_001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent_hl/models/Finetune_latent_hl_lr0.001_bs256_subset0.01.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent_hl/scores/0_01/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent_hl/models/Finetune_latent_hl_lr0.001_bs256_subset0.1.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent_hl/scores/0_1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent_hl/models/Finetune_latent_hl_lr0.001_bs256_subset1.0.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent_hl/scores/1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
##Scratch_latent
#config="/raven/u/mvigl/public/Finetune_hep/config/ParT_latent_config.yaml"
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent/models/Scratch_latent_lr0.001_bs256_subset0.0001.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent/scores/0_0001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent/models/Scratch_latent_lr0.001_bs256_subset0.001.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent/scores/0_001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent/models/Scratch_latent_lr0.001_bs256_subset0.01.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent/scores/0_01/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent/models/Scratch_latent_lr0.001_bs256_subset0.1.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent/scores/0_1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Scratch_latent/models/Scratch_latent_lr0.001_bs256_subset1.0.pt"
#out="/raven/u/mvigl/public/run/Scratch_latent/scores/1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#
##Finetune_latent
#config="/raven/u/mvigl/public/Finetune_hep/config/ParT_latent_config.yaml"
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent/models/Finetune_latent_lr0.001_bs256_subset0.0001.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent/scores/0_0001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent/models/Finetune_latent_lr0.001_bs256_subset0.001.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent/scores/0_001/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent/models/Finetune_latent_lr0.001_bs256_subset0.01.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent/scores/0_01/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent/models/Finetune_latent_lr0.001_bs256_subset0.1.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent/scores/0_1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h
#
#checkpoint="/raven/u/mvigl/public/run/Finetune_latent/models/Finetune_latent_lr0.001_bs256_subset1.0.pt"
#out="/raven/u/mvigl/public/run/Finetune_latent/scores/1/"
#nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
#disown -h





data="/raven/u/mvigl/public/Finetune_hep/config/train_list.txt"

#Finetune_Xbb_hl
config="/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml"
checkpoint="/raven/u/mvigl/public/run/Finetune_Xbb_hl/models/Finetune_Xbb_hl_lr0.001_bs256_subset1.0.pt"
out="/raven/u/mvigl/public/run/Finetune_Xbb_hl/scores/1/"
nohup sbatch --job-name="$mess" --time="${JOBTIME}" single_job_save_scores.sbatch "$config" "$checkpoint" "$mess" "$use_hlf" "$out" "$ishead" "$Xbb" "$repDim" "$data" "$out" >> "$LOG_FILE" 2>> "$ERR_FILE" &
disown -h
#exit 0
