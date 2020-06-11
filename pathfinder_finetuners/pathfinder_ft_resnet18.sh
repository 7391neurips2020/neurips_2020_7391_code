#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_resnet18_adam_400k -v1 resnet18 -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_marked_len16_resnet18_adam_400k_2020_04_20_02_37_42 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_resnet18_adam_400k -v1 resnet18 -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_resnet18_adam_400k_2020_05_09_21_10_56 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_resnet18_adam_400k -v1 resnet18 -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_resnet18_adam_400k_2020_05_10_04_24_43 -ft-dense True