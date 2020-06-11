#!/bin/bash

python main_estimator_markedlong.py -b 128 -ne 50 -lr 5e-4 -tr 1 -ob bce -op adam -en neurips_distractor_len16_ff_shallow_atrous_adam_400k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5
python main_estimator_markedlong.py -b 128 -ne 50 -lr 5e-4 -tr 1 -ob bce -op adam -en neurips_distractor_len16_ff_shallow_atrous_adam_400k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5
python main_estimator_markedlong.py -b 128 -ne 50 -lr 5e-4 -tr 1 -ob bce -op adam -en neurips_distractor_len16_ff_shallow_atrous_adam_400k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5
