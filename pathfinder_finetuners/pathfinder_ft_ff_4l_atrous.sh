#!/bin/bash


python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_4l_atrous_adam_400k -v1 ff_4l_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neurips_distractor_len16_ff_4l_atrous_adam_400k_2020_05_13_23_16_34 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_4l_atrous_adam_400k -v1 ff_4l_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neurips_distractor_len16_ff_4l_atrous_adam_400k_2020_05_14_02_34_14 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_4l_atrous_adam_400k -v1 ff_4l_atrous -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neurips_distractor_len16_ff_4l_atrous_adam_400k_2020_05_14_09_48_21 -ft-dense True
