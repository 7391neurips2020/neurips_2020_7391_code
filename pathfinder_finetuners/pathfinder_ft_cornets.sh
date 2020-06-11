#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_cornets_shallow_adam_400k -v1 cornets_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_k5_nipsrun_marked_len16_cornets_shallow_adam_400k_2020_05_13_04_38_37 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_cornets_shallow_adam_400k -v1 cornets_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_k5_nipsrun_marked_len16_cornets_shallow_adam_400k_2020_05_13_12_09_12 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_cornets_shallow_adam_400k -v1 cornets_shallow -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_k5_nipsrun_marked_len16_cornets_shallow_adam_400k_2020_05_13_19_40_04 -ft-dense True
