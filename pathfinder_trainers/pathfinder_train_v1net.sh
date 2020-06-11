#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_fastnipsrun_pathfinder_khw5_curv9_ft_fastnips_v1net_shallow_adam_400k -v1 v1net_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_marked_len16_v1net_shallow_adam_400k_2020_04_28_02_04_23 -ft-dense True

#python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_fastnipsrun_pathfinder_khw5_curv9_ft_fastnips_v1net_shallow_adam_400k -v1 v1net_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_v1net_shallow_adam_400k_2020_05_07_04_20_58 -ft-dense True

#python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_fastnipsrun_pathfinder_khw5_curv9_ft_fastnips_v1net_shallow_adam_400k -v1 v1net_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ft-dense True -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_v1net_shallow_adam_400k_2020_05_06_06_41_54
