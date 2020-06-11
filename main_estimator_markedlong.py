import argparse
import numpy as np
import tensorflow as tf
from models.openclose_model import openclose_model
from dataloaders.openclose_dataloader import  openclose_dataloader
import os
from utils.tf_utils import *
from absl import app
from utils.file_utils import *
import time

def add_args(parser):
    parser.add_argument('-ob','--obj',required=True)
    parser.add_argument('-en','--expt-name',required=True)
    parser.add_argument('-op','--opt',required=True)
    parser.add_argument('-b','--batch-sz',type=int,required=True)
    parser.add_argument('-dynamic','--dynamic-lr',required=False,default=False)
    parser.add_argument('-ne','--num-epochs',type=int,required=True)
    parser.add_argument('-lr','--learning-rate',type=float,required=True)
    parser.add_argument('-curv','--curv',type=int,required=True)
    parser.add_argument('-len','--len',type=int,required=True)
    parser.add_argument('-ts','--train-split',required=False,default='train')
    parser.add_argument('-tstp','--timesteps',required=False,default=5)
    parser.add_argument('-vs','--val-split',required=False,default='val')
    parser.add_argument('-trf','--train-pattern',type=str,required=False,default='train*')
    parser.add_argument('-valf','--val-pattern',type=str,required=False,default='val*')
    parser.add_argument('-v1','--v1-type', type=str, required=False, default='ff_binary')
    parser.add_argument('-kp','--keep-prob', type=float, required=False, default=0.9)
    parser.add_argument('-wd','--weight-decay', type=float, required=False, default=0)
    # eval freq specifies number of epochs per evaluation
    parser.add_argument('-eval-freq','--eval-freq', type=float, required=False, default=2.5)
    parser.add_argument('-tpu','--use-tpu',type=int,required=False,default=0)
    parser.add_argument('-ckpt','--ckpt',default=None)
    parser.add_argument('-tpu-name','--tpu-name',required=False)
    parser.add_argument('-ft-dense','--ft-dense',required=False)
    parser.add_argument('-test','--test',required=False,default=False)
    return parser

ws_scope = {'resnet50':'resnet*', 'resnet18': 'resnet*', 'resnet': 'resnet*',
             'ff_7l': 'ff_7l*', 'ff_4l': 'fft_4l*', 'v1net_shallow': 'shallow_v1*',
             'vgg16': 'vgg*', 'ff_smcnn': 'ff*',
             'v1net_bn_shallow': 'shallow_v1*',
             'ff_7l_atrous': 'ff_7l*', 'ff_4l_atrous': 'ff_4l*',
             'cornets_shallow': 'shallow_cornet*',
             'hgru_shallow': 'shallow_hgru*'
            }


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args_ = parser.parse_args()
    args = vars(args_)
    now = ret_datetime()
    base_path = 'gs://ADD_BUCKET_NAME_HERE'
    model_dir = '%s/open_close_tfrecords/output_dir_%s_%s'%(base_path,args['expt_name'],now)
    args['model_dir'] = model_dir
    rand_seed = np.random.randint(10000)
    tf.set_random_seed(rand_seed)
    args['random_seed'] = rand_seed
    if args['ckpt'] is not None:
        warm_start_from = args['ckpt']            
        warm_start_settings = tf.estimator.WarmStartSettings(            
                    ckpt_to_initialize_from=warm_start_from,
					vars_to_warm_start=[ws_scope[args['v1_type']]],
					)
    else:
        warm_start_settings = None
    if args['len'] == 16 or args['len'] == 18:
        train_tfr_pattern = '%s/openclose_data_1M/markedlong_len%s_distractor/openclose_tfrecords/train*.tfrecord'%(base_path, args['len']) #, args['curv'], args['len'])
        val_tfr_pattern = '%s/openclose_data_1M/markedlong_len%s_distractor/openclose_tfrecords/dev*.tfrecord'%(base_path, args['len']) #, args['curv'], args['len'])
        num_train_examples, num_val_examples, num_test_examples = 300000, 100000, 0
    openclose_train = openclose_dataloader(args['batch_sz'],train_tfr_pattern, image_size=256)
    openclose_val = openclose_dataloader(args['batch_sz'], val_tfr_pattern, training=False, image_size=256)
    args['num_train_examples'] = num_train_examples
    args['num_train_steps'] = num_train_steps = args['num_epochs'] * num_train_examples // args['batch_sz']
    args['num_train_steps_per_epoch'] = num_train_steps_per_epoch = num_train_examples // args['batch_sz']
    num_eval_steps = num_val_examples // args['batch_sz']
    eval_every = int(args['eval_freq'] * num_train_examples // args['batch_sz'])
    tf.logging.info('Evaluating every %s steps'%(eval_every))

    model = openclose_model(args)
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                  args['tpu_name'] if args['use_tpu'] else '',
                  zone='PROJECT_ZONE_REMOVED',
                  project='PROJECT_NAME_REMOVED')
    config = tf.compat.v1.estimator.tpu.RunConfig(
                  cluster=tpu_cluster_resolver,
                  model_dir=model_dir,
                  tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                                            # Since we use vx-8, i.e, 8 cores of vx tpu
                                            num_shards=8,
                                            iterations_per_loop=100))

    classifier = tf.compat.v1.estimator.tpu.TPUEstimator(
                  use_tpu=args['use_tpu'],
                  model_fn=model.model_fn,
                  config=config,
                  params=args,
                  warm_start_from=warm_start_settings,
                  train_batch_size=args['batch_sz'],
                  eval_batch_size=args['batch_sz'],
                  )

    # By default, this script always performs training and evaluation
    try:
        current_step = tf.train.load_variable(model_dir,
                                              tf.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
        current_step = 0
    start_timestamp = time.time()
    while current_step < num_train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + eval_every,
                              num_train_steps)
        if args['test']:
            tf.logging.info('Starting to evaluate %s.'%(args['ckpt']))
            eval_results = classifier.evaluate(
                            input_fn=openclose_val.input_fn,
                            steps=num_eval_steps,
                            )
            tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)
        classifier.train(
            input_fn=openclose_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(
                            input_fn=openclose_val.input_fn,
                            steps=num_eval_steps)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            num_train_steps, elapsed_time)

if __name__=='__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    app.run(main())
