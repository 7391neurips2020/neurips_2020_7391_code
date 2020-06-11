import numpy as np
import tensorflow as tf


def AdamOptim(objective_min,lr,global_step):
    adam = tf.train.AdamOptimizer(learning_rate=lr)
    grad_vars = adam.compute_gradients(objective_min)
    clipped_grad_vars = [(tf.clip_by_value(grad,-1.,1.),var)
                            for grad, var in grad_vars]
    step = adam.apply_gradients(clipped_grad_vars,global_step=global_step)
    return step


def RMSProp(objective_min,lr,global_step):
    print('Added RMSProp optimizer')
    rmsp = tf.train.RMSPropOptimizer(learning_rate=lr)
    grad_vars = rmsp.compute_gradients(objective_min)
    clipped_grad_vars = [(tf.clip_by_value(grad,-1.,1.),var)
                            for grad, var in grad_vars]
    step = rmsp.apply_gradients(clipped_grad_vars,global_step=global_step)
    return step


def SGD(objective_min,lr,global_step):
    sgd = tf.train.GradientDescentOptimizer(learning_rate=lr)
    grad_vars = sgd.compute_gradients(objective_min)
    clipped_grad_vars = [(tf.clip_by_value(grad,-1.,1.),var)
                            for grad, var in grad_vars]
    step = sgd.apply_gradients(clipped_grad_vars,global_step=global_step)
    return step


def Adagrad(objective_min,lr,global_step):
    agrad = tf.train.AdagradDAOptimizer(learning_rate=lr)
    grad_vars = agrad.compute_gradients(objective_min)
    clipped_grad_vars = [(tf.clip_by_value(grad,-1.,1.),var)
                            for grad, var in grad_vars]
    step = agrad.apply_gradients(clipped_grad_vars,global_step=global_step)
    return step


def get_or_create_global_step():
    """
    Checks if global step variable exists otherwise creates it
    :return:
    Global step tensor
    """
    global_step = tf.train.get_global_step()
    if global_step is None:
        global_step = tf.train.create_global_step()
    return global_step


def get_optimizer(lr, opt):
    if opt == 'adam':
        return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    elif opt == 'rmsprop':
        return tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr)
    elif opt == 'sgd':
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'momentum':
        return tf.train.MomentumOptimizer(
          			learning_rate=lr,
                                momentum=0.9,
          			use_nesterov=True) 


def get_lr_schedule(params):
    """learning rate schedule."""
    steps_per_epoch = params['num_train_steps_per_epoch']
    train_epochs = params['num_epochs']
    return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(1 / 10 * train_epochs)),
      (0.1, np.floor(2 / 10 * train_epochs)),
      (0.01, np.floor(3 / 10 * train_epochs)),
      (0.001, np.floor(4 / 10 * train_epochs))
    ]


def learning_rate_schedule(params, current_epoch):
    """Handles linear scaling rule, gradual warmup, and LR decay.
    The learning rate starts at 0, then it increases linearly per step.
    After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.
    Args:
    params: Python dict containing parameters for this run.
    current_epoch: `Tensor` for current epoch.
    Returns:
    A scaled `Tensor` for current learning rate.
    """
    scaled_lr = params['learning_rate'] * (
            params['batch_sz'] / 256.0)
    lr_schedule = get_lr_schedule(params)
      # train_steps=params['num_train_steps'],
      # num_train_images=params['num_train_examples'],
      # train_batch_size=params['batch_sz'])

    decay_rate = (scaled_lr * lr_schedule[0][0] *
                current_epoch / lr_schedule[0][1])
    for mult, start_epoch in lr_schedule:
        decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
    return decay_rate


def add_optimizer(objective_min, lr, opt, use_tpu=True):
    """Function to add an mizer
     and return the update op
     Exponential learning rate decay to prevent loss oscillation"""
    if use_tpu:
        #TODO : Add learning rate scheduler
        print('Adding optimizer',opt)
        optim = get_optimizer(lr, opt)
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optim)
        return optimizer
        
    global_step = get_or_create_global_step()
    starter_learning_rate = lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               1000, 0.96,
                                               staircase=True)
    # Passing global_step to minimize() will increment it at each step.
    if opt == 'adam':
        step = AdamOptim(
                        objective_min,
                        lr=learning_rate,
                        global_step=global_step,
                        )
    if opt == 'rmsprop':
        step = RMSProp(
                        objective_min,
                        lr=learning_rate,
                        global_step=global_step,
                        )
    if opt == 'sgd':
        step = SGD(
                        objective_min,
                        lr=learning_rate,
                        global_step=global_step,
                        )
    if opt== 'adagrad':
        step = Adagrad(
                        objective_min,
                        lr=learning_rate,
                        global_step=global_step,
                        )
    return step
