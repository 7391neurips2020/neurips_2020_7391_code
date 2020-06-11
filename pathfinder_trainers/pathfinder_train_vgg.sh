#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 10 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_pathfinder_curv9_vgg_adam_400k -v1 vgg -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckptgs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_vgg16_adam_400k_2020_05_09_21_06_01 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 10 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_pathfinder_curv9_vgg_adam_400k -v1 vgg -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_marked_len16_vgg16_adam_400k_2020_04_21_00_13_20 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 10 -lr 5e-4 -tr 1 -ob bce -op adam -en nipsrun_pathfinder_curv9_vgg_adam_400k -v1 vgg -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_vgg16_adam_400k_2020_05_10_13_29_38 -ft-dense True
