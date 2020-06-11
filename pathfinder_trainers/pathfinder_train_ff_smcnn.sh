#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 10 -lr 5e-4 -tr 1 -ob bce -op adam -en samepad_khw5_pathfinder_curv9_fastnips_ff_smcnn_adam_100k -v1 ff_smcnn -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en samepad_khw5_pathfinder_curv9_fastnips_ff_smcnn_adam_100k -v1 ff_smcnn -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en samepad_khw5_pathfinder_curv9_fastnips_ff_smcnn_adam_100k -v1 ff_smcnn -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5
