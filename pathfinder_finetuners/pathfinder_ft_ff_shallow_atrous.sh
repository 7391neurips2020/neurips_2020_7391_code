#!/bin/bash


python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_shallow_atrous_adam_400k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neurips_distractor_len16_ff_shallow_atrous_adam_400k_2020_05_22_22_47_12 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_shallow_atrous_adam_400k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neurips_distractor_len16_ff_shallow_atrous_adam_400k_2020_05_23_02_12_15 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_shallow_atrous_adam_400k -v1 ff_shallow_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neurips_distractor_len16_ff_shallow_atrous_adam_400k_2020_05_23_05_37_13 -ft-dense True
