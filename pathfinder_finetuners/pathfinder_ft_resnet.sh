#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en transfernipsft_pathfinder_neurips_mltopf_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_marked_len16_resnet50_adam_400k_2020_05_02_05_42_29 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en transfernipsft_pathfinder_neurips_mltopf_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_resnet50_adam_400k_2020_05_09_21_04_20 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en transfernipsft_pathfinder_neurips_mltopf_resnet_adam_400k -v1 resnet -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_nipsrun_marked_len16_resnet50_adam_400k_2020_05_10_04_37_16 -ft-dense True
