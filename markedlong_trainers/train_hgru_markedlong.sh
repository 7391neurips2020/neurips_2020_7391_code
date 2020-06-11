#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/NAME_REMOVED/src/v1net_master/:/home/NAME_REMOVED/src/v1net_master/hgru_share

python main_estimator_markedlong.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_marked_len16_hgru_shallow_adam_400k -v1 hgru_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -curv 2 -len 16 -kp 0.5
python main_estimator_markedlong.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_marked_len16_hgru_shallow_adam_400k -v1 hgru_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -curv 2 -len 16 -kp 0.5
python main_estimator_markedlong.py -b 128 -ne 25 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_marked_len16_hgru_shallow_adam_400k -v1 hgru_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -curv 2 -len 16 -kp 0.5
