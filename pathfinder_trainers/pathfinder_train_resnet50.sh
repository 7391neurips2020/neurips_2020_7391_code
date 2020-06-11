#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_pathfinder_curv9_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_pathfinder_curv9_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_pathfinder_curv9_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5