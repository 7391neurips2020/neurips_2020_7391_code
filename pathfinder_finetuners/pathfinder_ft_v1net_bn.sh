#!/bin/bash

#python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_curv9_neurips_mltopf_v1net_bn_shallow_adam_400k -v1 v1net_bn_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_final_nipsrun_marked_len16_v1net_bn_shallow_adam_400k_2020_06_02_06_29_53 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_curv9_neurips_mltopf_v1net_bn_shallow_adam_400k -v1 v1net_bn_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_final_nipsrun_marked_len16_v1net_bn_shallow_adam_400k_2020_06_03_20_30_10 -ft-dense True

#python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_curv9_neurips_mltopf_v1net_bn_shallow_adam_400k -v1 v1net_bn_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_final_nipsrun_marked_len16_v1net_bn_shallow_adam_400k_2020_06_02_06_33_14 -ft-dense True
