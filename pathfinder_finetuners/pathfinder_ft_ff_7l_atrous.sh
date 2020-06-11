#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en nipstransfer_pathfinder_khw5_curv9_fastnips_ff_7l_atrous_adam_100k -v1 ff_7l_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_7l_atrous_adam_400k_2020_05_14_14_07_04 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en nipstransfer_pathfinder_khw5_curv9_fastnips_ff_7l_atrous_adam_100k -v1 ff_7l_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_7l_atrous_adam_400k_2020_05_14_06_42_20 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 15 -lr 5e-4 -tr 1 -ob bce -op adam -en nipstransfer_pathfinder_khw5_curv9_fastnips_ff_7l_atrous_adam_100k -v1 ff_7l_atrous -tpu 1 -tpu-name $1 -eval-freq 1 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_7l_atrous_adam_400k_2020_05_13_23_10_33 -ft-dense True
