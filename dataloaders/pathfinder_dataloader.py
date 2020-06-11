import tensorflow as tf
import numpy as np
import multiprocessing
from utils.tf_utils import augment_images
import functools

class pathfinder_dataloader(object):
    def __init__(self, batch_size, file_pattern, training=True):
        self.batch_size = batch_size
        self.file_pattern = file_pattern

        self.training = training

    def decode_feats(self, tfrecord):
        # Function to decode features written in
        # tfrecords using feature dictionary
        if self.training:
            split = 'train'
        else:
            split = 'dev'
        feat_dict = {
            '%s/shape' % (split): tf.FixedLenFeature(
                [2], tf.int64),
            '%s/image' % (split): tf.FixedLenFeature(
                [], tf.string),
            '%s/label' % (split): tf.FixedLenFeature(
                [], tf.int64),
            '%s/file_id'%(split): tf.FixedLenFeature(
                [], tf.int64),
                        }
        sample = tf.parse_single_example(tfrecord, feat_dict)
        # decoding shape
        shape = sample['%s/shape' % (split)]
        # decoding image
        img = tf.decode_raw(sample['%s/image' % (split)], tf.uint8)
        img = tf.reshape(img, [300, 300, 3])
        img = tf.image.resize(
		            img, [150,150]
                )

        img = tf.cast(img, tf.float32)
        img = img/255.
        if self.training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
        img = tf.clip_by_value(img, 0., 1.)
        # decoding label (class indicator)
        label = sample['%s/label' % (split)]
        file_id = sample['%s/file_id' % (split)]
        # label = tf.one_hot(label, depth=2)
        # TODO: Add data augmentation here
        return {'image': img, 'file_id': file_id}, {'label': label}

    def set_shapes(self, batch_size, images, labels):
        """Statically set the batch_size dimension."""
        images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size,2])))

    def define_batch_size(self, features, labels):
        """
        Define batch size of feature and label dictionary
        :param features: Features dict with image tensor
        :param labels: Labels dict with boundary map GT tensor
        :return: tuple with features and labels
        """
        for key, val in features.iteritems():
            features[key] = tf.reshape(val, [self.batch_size] + val.shape.as_list()[1:])

        for key, val in labels.iteritems():
            labels[key] = tf.reshape(val, [self.batch_size] + val.shape.as_list()[1:])

    def read_tf_records(self,
                         glob_pattern,
                        ):
        """Function to read tfrecords for open vs close
        :param glob_pattern: pattern used to match tfrecord filenames
        """

        threads = multiprocessing.cpu_count()

        # load tfrecord files
        files = tf.data.Dataset.list_files(glob_pattern, shuffle=self.training)

        # parallel fetching of tfrecords dataset
        dataset = files.apply(tf.data.experimental.parallel_interleave(
                                lambda filename: tf.data.TFRecordDataset(filename),
                                cycle_length=threads, sloppy=True))

        # shuffling dataset
        # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(70000))
        dataset = dataset.shuffle(
                  buffer_size=32*self.batch_size, reshuffle_each_iteration=True).repeat()

        # use decode function to retrieve images and labels
        dataset = dataset.apply(
                    tf.data.experimental.map_and_batch(self.decode_feats,
                                                        batch_size=self.batch_size,
                                                        num_parallel_batches=threads,
                                                        drop_remainder=True))
        # dataset = dataset.map(map_func=lambda example: decode_feats(example, self.training),
        #                       num_parallel_calls=threads)
        # dataset = dataset.map(functools.partial(self.set_shapes, self.batch_size))
        # batch the examples
        # dataset = dataset.batch(batch_size=self.batch_size)

        # prefetch batch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        #return dataset.make_one_shot_iterator().get_next()
        return dataset

    def input_fn(self, params):
        """
        Input function to feed data to TPU model
        :param params: parameter dictionary containing experimental hyperparameters
        :return:
        """
        # return self.define_batch_size(self.read_tf_records(self.file_pattern))
        return self.read_tf_records(self.file_pattern)

