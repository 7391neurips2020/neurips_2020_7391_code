#!/bin/bash


python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_9l_adam_400k -v1 ff_9l -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_9l_adam_400k_2020_05_22_06_26_14 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_9l_adam_400k -v1 ff_9l -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_9l_adam_400k_2020_05_22_06_47_40 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en ft_pathfinder_neurips_mltopf_ff_9l_adam_400k -v1 ff_9l -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_9l_adam_400k_2020_05_22_06_26_08 -ft-dense True
