#!/bin/bash

python main_estimator_pathfinder.py -b 128 -ne 10 -lr 5e-4 -tr 1 -ob bce -op adam -en samepad_khw5_pathfinder_curv9_fastnips_ff_smcnn_adam_100k -v1 ff_smcnn -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_smcnn_adam_400k_2020_05_13_17_26_51 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en samepad_khw5_pathfinder_curv9_fastnips_ff_smcnn_adam_100k -v1 ff_smcnn -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_smcnn_adam_400k_2020_05_13_10_00_07 -ft-dense True

python main_estimator_pathfinder.py -b 128 -ne 20 -lr 5e-4 -tr 1 -ob bce -op adam -en samepad_khw5_pathfinder_curv9_fastnips_ff_smcnn_adam_100k -v1 ff_smcnn -tpu 1 -tpu-name $1 -eval-freq 0.5 -len 9 -kp 0.5 -ckpt gs://ADD_BUCKET_NAME_HERE/open_close_tfrecords/output_dir_neuripsrun_marked_len16_ff_smcnn_adam_400k_2020_05_13_02_21_14 -ft-dense True
