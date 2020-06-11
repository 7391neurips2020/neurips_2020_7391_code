#!/bin/bash

python main_estimator_markedlong.py -b 128 -ne 50 -lr 7e-5 -tr 1 -ob bce -op adam -en marked_len16_ff_9l_adam_400k -v1 ff_9l -tpu 1 -tpu-name $1 -eval-freq .25 -curv 2 -len 16 -kp 0.5
