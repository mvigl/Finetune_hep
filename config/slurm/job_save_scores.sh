#!/bin/bash

config="$1"
checkpoint="$2"
mess="$3"
use_hlf="$4"
out="$5"
ishead="$6"
Xbb="$7"
repDim="$8"
data="$9"

python /raven/u/mvigl/public/Finetune_hep/save_scores.py --config "$config" --checkpoint "$checkpoint" --Xbb "$Xbb" --repDim "$repDim" --data "$data" --out "$out" #--ishead --use_hlf #--save_representaions #--ishead  #"$use_hlf" "$save_representaions" "$ishead"