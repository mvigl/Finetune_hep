#!/bin/bash

config="$1"
checkpoint="$2"
mess="$3"
data="$4"
subset="$5"
data_val="$6"
out="$7"
Xbb="$8"
Xbb_val="$9"

python /raven/u/mvigl/public/Finetune_hep/training_frozen.py --config "$config" --checkpoint "$checkpoint" --mess "$mess" --data "$data" --subset "$subset" --data_val "$data_val" --out "$out" --Xbb "$Xbb" --Xbb_val "$Xbb_val" --bs 256
