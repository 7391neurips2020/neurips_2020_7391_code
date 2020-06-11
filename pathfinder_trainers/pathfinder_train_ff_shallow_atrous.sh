#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 1e-3 -tr 1 -ob bce -op adam -en fastnips_samepad_pathfinder_curv9_ff_shallow_atrous_adam_100k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 1e-3 -tr 1 -ob bce -op adam -en fastnips_samepad_pathfinder_curv9_ff_shallow_atrous_adam_100k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 1e-3 -tr 1 -ob bce -op adam -en fastnips_samepad_pathfinder_curv9_ff_shallow_atrous_adam_100k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5
