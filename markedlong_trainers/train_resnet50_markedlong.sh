#!/bin/bash

python main_estimator_markedlong.py -b 128 -ne 15 -lr 1e-4 -tr 1 -ob bce -op adam -en transfernipsrun_marked_len16_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5

python main_estimator_markedlong.py -b 128 -ne 15 -lr 1e-4 -tr 1 -ob bce -op sgd -en transfernipsrun_marked_len16_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5

python main_estimator_markedlong.py -b 128 -ne 15 -lr 1e-4 -tr 1 -ob bce -op adam -en transfernipsrun_marked_len16_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5
