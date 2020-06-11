#!/bin/bash

python main_estimator_markedlong.py -b 128 -ne 50 -lr 5e-3 -tr 1 -ob bce -op adam -en nipsrun_marked_len16_vgg16_adam_400k -v1 vgg16 -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5 -ft-dense True
python main_estimator_markedlong.py -b 128 -ne 50 -lr 5e-3 -tr 1 -ob bce -op adam -en nipsrun_marked_len16_vgg16_adam_400k -v1 vgg16 -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5 -ft-dense True
python main_estimator_markedlong.py -b 128 -ne 50 -lr 5e-3 -tr 1 -ob bce -op adam -en nipsrun_marked_len16_vgg16_adam_400k -v1 vgg16 -tpu 1 -tpu-name $1 -eval-freq 1 -curv 2 -len 16 -kp 0.5 -ft-dense True
