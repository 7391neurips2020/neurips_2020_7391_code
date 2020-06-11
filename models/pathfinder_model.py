import tensorflow as tf
import numpy as np
from dataloaders.pathfinder_dataloader import pathfinder_dataloader
from utils.tf_utils import *
from utils import obj
from utils import opt
from utils.opt import *
import os
from models.v1net_shunt_model import *
from models.resnet_model import *


class pathfinder_model(object):
    def __init__(self, args):
        self.batch_size = args['batch_sz']
        self.num_epochs = args['num_epochs']
        self.obj = args['obj']
        self.opt = args['opt']
        self.expt_name = args['expt_name']
        self.use_tpu = args['use_tpu']

    def model_fn(self, features, labels, mode, params):
        """
        Function to create the pathfinder classification models
        :param features: Features/images tensor
        :param labels: Labels/GT tensor
        :param mode: train/eval/predict flag
        :param params: dict of configuration parameters
        :return:
        """
        training = mode == tf.estimator.ModeKeys.TRAIN
        train_steps = tf.cast(params['num_train_steps'], tf.float32)
        current_step = tf.cast(tf.train.get_global_step(),tf.float32)
        current_ratio = current_step / train_steps
        dropout_keep_prob = params['keep_prob'] # (1 - current_ratio * (1 - params['keep_prob']))
        if training:
            keep_prob = dropout_keep_prob
        else:
            keep_prob = tf.constant(1.)
        # create model inside the argscope of the model
        if type(features) == dict:
            if params['v1_type'] == 'ff_binary':
                logits, model_params = ff_binary(features['image'],
                                                 labels['label'],
                                                 training=training,
                                                 keep_prob=keep_prob,
                                                 num_classes=2)
            elif params['v1_type'] == 'ff_smcnn':                         
                logits, model_params = ff_7l_smcnn(features['image'],
                                                    labels['label'],
                                                    training=training,
                                                    keep_prob=keep_prob,
                                                    num_classes=2)
            elif params['v1_type'] == 'ff_9l':
                logits, model_params = ff_9l(features['image'],
                                                labels['label'],
                                                training=training,
                                                keep_prob=keep_prob,
                                                num_classes=2)
            elif params['v1_type'] == 'ff_shallow':
                logits, model_params = ff_shallow(features['image'],
                                                labels['label'],
                                                training=training,
                                                keep_prob=keep_prob,
                                                num_classes=2)
            elif params['v1_type'] == 'ff_7l_large':
                logits, model_params = ff_7l_large(features['image'],
                                                labels['label'],
                                                training=training,
                                                keep_prob=keep_prob,
                                                num_classes=2)
            elif params['v1_type'] == 'ff_4l':
                logits, model_params = ff_4l(features['image'],
                                            labels['label'],
                                            training=training,
                                            keep_prob=keep_prob,
                                            num_classes=2)
            elif params['v1_type'] == 'ff_4l_atrous':
                logits, model_params = ff_4l_atrous(features['image'],
                                            labels['label'],
                                            training=training,
                                            keep_prob=keep_prob,
                                            num_classes=2)
            elif params['v1_type'] == 'ff_7l':
                logits, model_params = ff_7l(features['image'],
                                             labels['label'],
                                             training=training,
                                             keep_prob=keep_prob,
                                             num_classes=2)
            elif params['v1_type'] == 'ff_9l_atrous':
                logits, model_params = ff_9l_atrous(features['image'],
                                                labels['label'],
                                                training=training,
                                                keep_prob=keep_prob,
                                                num_classes=2)
            elif params['v1_type'] == 'ff_shallow_atrous':
                logits, model_params = ff_shallow_atrous(features['image'],
                                                    labels['label'],
                                                    training=training,
                                                    keep_prob=keep_prob,
                                                    num_classes=2,
                                                    )
            elif params['v1_type'] == 'ff_7l_atrous':
                logits, model_params = ff_7l_atrous(features['image'],
                                                    labels['label'],
                                                    training=training,
                                                    keep_prob=keep_prob,
                                                    num_classes=2)
            elif params['v1_type'] == 'vgg':
                logits, model_params = vgg16(features['image'],
                                             labels['label'],
                                             training=training,
                                             keep_prob=keep_prob,
                                             num_classes=2)
            elif 'resnet' in params['v1_type']:
                depth = 50
                if '18' in params['v1_type']:
                    depth = 18
                with tf.variable_scope('resnet_model'):
                    logits, model_params = resnet_v2(features['image'],
                                                labels['label'],
                                                is_training=training,
                                                num_classes=2,
                                                resnet_depth=depth
                                                )
            elif params['v1_type'] == 'cornets_shallow':
                logits, model_params = cornets_shallow(features['image'],
                                                 labels['label'],
                                                 training=training,
                                                 keep_prob=keep_prob,
                                                 num_classes=2)
            
            elif params['v1_type'] == 'hgru_shallow':                                                       
                logits, model_params = hgru_shallow(features['image'],                   
                                                    labels['label'],                         
                                                    training=training,                         
                                                    keep_prob=keep_prob,         
                                                    num_classes=2)

            elif params['v1_type'] == 'v1net_nl':
                logits, model_params = v1net_nl(features['image'],
                                                labels['label'],
                                                training,
                                                keep_prob=keep_prob,
                                                num_classes=2)
            elif params['v1_type'] == 'v1net_lstm':
                logits, model_params = v1net_nl(features['image'],
                                                labels['label'],
                                                training, keep_prob=keep_prob,
                                                num_classes=2)
            elif params['v1_type'] == 'v1net_shallow':
                logits, model_params = v1net_shallow(features['image'],
                                                 labels['label'],
                                                 training=training,
                                                 keep_prob=keep_prob,
                                                 num_classes=2)
            elif params['v1_type'] == 'v1net_bn_shallow':
                logits, model_params = v1net_bn_shallow(features['image'],
                                                 labels['label'],
                                                 training=training,
                                                 keep_prob=keep_prob,
                                                 num_classes=2)
            elif params['v1_type'] == 'v1net_shallow_bf16':
                with tf.tpu.bfloat16_scope():
                    logits, model_params = v1net_shallow(features['image'],
                                                 labels['label'],
                                                 training=training,
                                                 keep_prob=keep_prob,
                                                 num_classes=2)
                logits = tf.cast(logits, tf.float32)
            elif params['v1_type'] == 'v1net_middle_nl':
                logits, model_params = v1net_middle_nl(features['image'],
                                                 labels['label'],
                                                 training=training,
                                                 keep_prob=keep_prob,
                                                 num_classes=2)
        else:
            raise(TypeError, 'Incompatible features/label type')
        # output predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            sigmoid = 1/(1+tf.exp(logits))
            predictions = {
                'logits': logits,
                'probabilities': sigmoid,
                }
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

        # create loss (should be equal to caffe softmaxwithloss)
        loss_ce = tf.losses.sparse_softmax_cross_entropy(
                      logits=logits,
                      labels=labels['label'],
                      )
                  
        #loss_wd =  params['weight_decay'] * tf.add_n([
        #            tf.nn.l2_loss(v) for v in tf.trainable_variables()
        #            if 'batch_normalization' not in v.name])
        loss = loss_ce 
        host_call = None
        eval_hook = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            # Added by Author1 for grad image
            #dx = tf.gradients(loss, features['image'])
            current_epoch = tf.cast(global_step, tf.float32)/params['num_train_steps']
            learning_rate = params['learning_rate'] # learning_rate_schedule(params, current_epoch)
            vars_to_train = [var for var in tf.trainable_variables()]
            print('%s'%(len(vars_to_train)),'variables to train')
            if params['ft_dense']:
                print('$'*30)
                vars_to_train = [var for var in tf.trainable_variables() 
                                        if 'linear' in var.name or 'readout' in var.name or 'dense' in var.name or 'batch_norm' in var.name or 'layer_norm' in var.name]
                print('%s'%(len(vars_to_train)),'variables to train')
            optimizer = opt.add_optimizer(loss, learning_rate, self.opt, self.use_tpu)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = optimizer.minimize(loss, global_step, var_list = vars_to_train)
            train_op = tf.group([train_op, update_ops])
            #if 'cornet' in params['v1_type']:
            #    with tf.control_dependencies(update_ops):
            #        train_op = optimizer.minimize(loss, global_step, var_list=vars_to_train)
            #else:
            #    train_op = optimizer.minimize(loss, global_step, var_list=vars_to_train)
            #    train_op = tf.group([train_op, update_ops])
            #dl_dx = tf.gradients(loss, features['image'])[0]

            def normalize_gradients(dx):
                """This function normalizes gradient tensors for visualization
                :param dx: gradient of loss wrt input images"""
                dx_min = tf.math.reduce_min(dx, axis=(1,2,3), keepdims=True)
                dx_max = tf.math.reduce_max(dx, axis=(1,2,3), keepdims=True)
                norm = (dx-dx_min)/(dx_max-dx_min)
                return norm

            def host_call_fn(gs, loss, loss_ce, lr, img, kp, file_id): # dx, file_id):
                """Training host call. Creates scalar summaries for training metrics.
                This function is executed on the CPU and should not directly reference
                any Tensors in the rest of the `model_fn`. To pass Tensors from the
                model to the `metric_fn`, provide as part of the `host_call`. See
                https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
                for more information.
                Arguments should match the list of `Tensor` objects passed as the second
                element in the tuple passed to `host_call`.
                Args:
                  gs: `Tensor with shape `[batch]` for the global_step
                  loss: `Tensor` with shape `[batch]` for the training loss.
                  lr: `Tensor` with shape `[batch]` for the learning_rate.
                Returns:
                  List of summary ops to run on the CPU host.
                """
                gs = gs[0]

                # Host call fns are executed params['iterations_per_loop'] times after
                # one TPU loop is finished, setting max_queue value to the same as
                # number of iterations will make the summary writer only flush the data
                # to storage once per loop.
                with tf.compat.v2.summary.create_file_writer(
                        params['model_dir'],
                        max_queue=32).as_default():
                    with tf.compat.v2.summary.record_if(True):
                        #for var in tf.trainable_variables():
                        #    tf.compat.v2.summary.histogram('weights/%s'%(var.name), var, step=gs)
                        tf.compat.v2.summary.histogram('training/file_id', file_id, step=gs)
                        tf.compat.v2.summary.scalar('training/cross_entopy',loss_ce[0], step=gs)
                        #tf.compat.v2.summary.scalar('training/weight_decay',loss_wd[0], step=gs)
                        #tf.compat.v2.summary.scalar('training/loss', loss[0], step=gs)
                        tf.compat.v2.summary.scalar('training/learning_rate', lr[0], step=gs)
                        tf.compat.v2.summary.scalar('training/keep_prob', kp[0], step=gs)
                        #tf.compat.v2.summary.image('training/images',img,step=gs)
                        tf.compat.v2.summary.text('training/model_params',str(model_params),step=0)
                        tf.compat.v2.summary.text('training/training_params',str(params),step=0)
                        #tf.compat.v2.summary.image('training/dx',dx,step=gs)
                        #tf.compat.v2.summary.image('training/dl_close',dl_close,step=gs)
                        #tf.compat.v2.summary.image('training/dl_open_avg',open_mean,step=gs)
                        #tf.compat.v2.summary.image('training/dl_close_avg',close_mean,step=gs)
                    return tf.summary.all_v2_summary_ops()

                # To log the loss, current learning rate, and epoch for Tensorboard, the
                # summary op needs to be run on the host CPU via host_call. host_call
                # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
                # dimension. These Tensors are implicitly concatenated to
                # [params['batch_size']].
            #gs_t = tf.reshape(global_step, [1])
            #loss_t = tf.reshape(loss, [1])
            #lr_t = tf.reshape(learning_rate, [1])
            #img_t = features['image']

            #dx_t = normalize_gradients(dx[0]) 
            #kp_t = tf.reshape(keep_prob, [1])
            #loss_ce_t = tf.reshape(loss_ce, [1])
            #loss_wd_t = tf.constant(0.) #tf.reshape(loss_wd, [1])
            #file_id_t = features['file_id']
            scaffold = tf.train.Scaffold()
            #host_call = (host_call_fn, [gs_t, loss_t, loss_ce_t, #loss_wd_t, 
            #                            lr_t, img_t, kp_t, #dl_open_t, 
                                        #dl_close_t, 
                                        #dx_t,
            #                            file_id_t])
        else:
            train_op = None

        eval_metrics=None
        if mode == tf.estimator.ModeKeys.EVAL:
            # Define the metrics:
            def metric_fn(labels_fun, logits_fun):
                probs = tf.nn.softmax(logits_fun)
                predictions = tf.argmax(probs, axis=1)
                accuracy = tf.compat.v1.metrics.accuracy(labels=labels_fun, 
                                                        predictions=predictions)
                return {
                        'accuracy': accuracy,
                        }
            label_tensor = labels['label']
            eval_metrics = (metric_fn, [label_tensor, logits])
        return tf.estimator.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=loss,
                        train_op=train_op,
                        host_call=host_call,
                        eval_metrics=eval_metrics,
                        )
