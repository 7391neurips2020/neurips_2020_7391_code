#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 25 -lr 1e-3 -tr 1 -ob bce -op adam -en pathfinder_khw5_curv9_fastnips_ff_4l_atrous_adam_100k -v1 ff_4l_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 25 -lr 1e-3 -tr 1 -ob bce -op adam -en pathfinder_khw5_curv9_fastnips_ff_4l_atrous_adam_100k -v1 ff_4l_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 25 -lr 1e-3 -tr 1 -ob bce -op adam -en pathfinder_khw5_curv9_fastnips_ff_4l_atrous_adam_100k -v1 ff_4l_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5
