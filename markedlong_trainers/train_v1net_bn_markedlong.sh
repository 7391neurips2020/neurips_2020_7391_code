#!/bin/bash

python main_estimator_markedlong.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en final_nipsrun_marked_len16_v1net_bn_shallow_adam_400k -v1 v1net_bn_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -curv 2 -len 16 -kp 0.1
python main_estimator_markedlong.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en final_nipsrun_marked_len16_v1net_bn_shallow_adam_400k -v1 v1net_bn_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -curv 2 -len 16 -kp 0.5
python main_estimator_markedlong.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en final_nipsrun_marked_len16_v1net_bn_shallow_adam_400k -v1 v1net_bn_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -curv 2 -len 16 -kp 0.5
