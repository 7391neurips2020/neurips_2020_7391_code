#!/bin/bash


python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_4l_adam_400k -v1 ff_4l -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_samepad_neurips_distractor_len16_ff_4l_adam_400k_2020_05_11_02_24_09 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_4l_adam_400k -v1 ff_4l -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_samepad_neurips_distractor_len16_ff_4l_adam_400k_2020_05_11_17_33_35 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_4l_adam_400k -v1 ff_4l -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_samepad_neurips_distractor_len16_ff_4l_adam_400k_2020_05_12_08_47_02 -ft-dense True
