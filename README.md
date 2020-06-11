# Code for Paper ID 7391
This directory contains code accompanying Paper ID 7391's submission to NeurIPS 2020.
All our code is written in Python 3.7.4, we use the TensorFlow framework to implement our deep learning models.

All bash scripts used for training our 15 feedforward and recurrent baselines on MarkedLong are available in markedlong_trainers.
For a new training run, please copy/move the required training script from markedlong_trainers to the base repository and run the following command:

Eg. to train FF-1L model on MarkedLong on tpu node tpu1:

`bash train_ff_shallow_markedlong.sh tpu1`

Eg. to finetune an FF-1L model on tpu node tpu2:

`bash pathfinder_ft_ff_shallow.sh tpu2`

Eg. to train FF-1L model on PathFinder on tpu node tpu3:

`bash pathfinder_train_ff_shallow.sh tpu3`

In order to reproduce experimental results from this paper, copy all bash scripts from markedlong_trainers, pathfinder_trainers, and pathfinder_finetuners to base path and run them on TPUs. 
All data should be stored in GCS buckets, GCS project name should be mentioned in main_estimator_*.py 

All python package requirements are stored in v1net_tpu_requirements.txt

Description of folders and files:
- dataloaders: Reading from and writing to tfrecords
    - open_close_datawriter.py  -- Contains python code for writing markedlong data to tfrecords
    - openclose_dataloader.py -- Contains python code to read markedlong data from tfrecords
    - path_finder_datawriter.py --  Contains python code for writing pathfinder data to tfrecords
    - pathfinder_dataloader.py -- Contains python code to read pathfinder data from tfrecords
   
- models: Contains code to generate all comparison models for training on MarkedLong and on PathFinder
    - openclose_model.py -- Contains python code to instantiate model, loss and optimizers for training
    on MarkedLong
    - pathfinder_model.py -- Contains python code to instantiate model, loss and optimizers for training
    on PathFinder
    - resnet_model.py -- Original tensorflow implementation of all ResNet_v2 models.
    - v1net_shunt_model.py -- Contains python code to generate all of the first 13 models (except ResNet-18, ResNet-50)
    from paper ID 7391
    - vgg_model.py -- Contains TensorFlow slim implementation of VGG architectures without BatchNormalization.

- layers: Contains code to implement various layers used in our project in TensorFlow
    - conv_rnn: Contains code to convolutional layer and recurrent layer implementations
        - conv_layers: implements Conv2D, atrous_conv2d, etc.
        - rnn_layers: implements V1Net layer, CORNet-S layer, etc.
    - horizontal_cells:
        - hgru_cell.py: Actually implements GRU-1L in the paper
        - horizontal_lstm_linear_ei: Implements a linear variant of V1Net
        - v1net_bn_cell.py: Implements V1Net recurrent cell used in the paper with LayerNorm
        - v1net_cell.py: Implementation of V1Net without normalization
    - readouts: Contains code to implement different readouts, not used for submission 7391

- markedlong_trainers:
    - Contains all bash scripts to train models from scratch on MarkedLong
    
- pathfinder_finetuners:
    - Contains all bash scripts to finetune models from MarkedLong to PathFinder
    
- pathfinder_trianers:
    - Contains all bash scripts to train models from scratch on PathFinder
    
- utils: Miscellaneous utility scripts
    - file_utils.py -- functions to create directories, delete directories, parse json files etc.
    - obj.py -- functions to implement various classification and regression objective functions
    - opt.py -- functions to instantiate various optimization algorithms
    - plotter_utils.py -- helper functions for matplotlib plotting
    - regularizers.py -- functions to implement various regularization techniques
    - resnet_preprocessing.py -- preprocessor functions for training ResNet models
    - tf_utils.py -- Miscellaneous tensorflow helper functions, i.e, various layer implementations such as conv2D, atrous2D, Dense layers etc.
    
- main_estimator_markedlong.py:
    - Main training script for MarkedLong experiments
   
- main_estimator_pathfinder.py:
    - Main training script for PathFinder experiments and transfer learning experiments