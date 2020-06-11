#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 1e-3 -tr 1 -ob bce -op adam -en fastnipsrun_pathfinder_khw5_curv9_fastnips_hgru_shallow_adam_400k -v1 hgru_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 1e-3 -tr 1 -ob bce -op adam -en fastnipsrun_pathfinder_khw5_curv9_fastnips_hgru_shallow_adam_400k -v1 hgru_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 1e-3 -tr 1 -ob bce -op adam -en fastnipsrun_pathfinder_khw5_curv9_fastnips_hgru_shallow_adam_400k -v1 hgru_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5
