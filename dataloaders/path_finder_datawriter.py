import numpy as np
import cv2
import tensorflow as tf
from os.path import join
import scipy.io as sio
from utils.file_utils import *
import glob
import sys
from tqdm import tqdm
from dataloaders.pathfinder_dataloader import pathfinder_dataloader

class pathfinder:
    def __init__(
            self,
            data_path,
            img_shape_xy,
            tfr_path,
            shuffle,
            im_ext,
            tfr_ext,
            write_train=True,
            write_dev=True,
            ):
        self.data_path = data_path
        # Checking the sync
        self.tfr_path = join(data_path,'pathfinder_%s'%(tfr_path))
        self.img_shape_xy = img_shape_xy
        self.im_ext = im_ext
        self.tfr_ext = tfr_ext
        # Closed: 1, Open: 0
        self.classes = ['closed', 'open']
        self.shuffle = True
        self.px_min = 0.
        self.px_max = 255.
        self.write_train = write_train
        self.write_dev = write_dev
        self.create_train_test()

    def create_train_test(self):
        """Function to return image
        names from the given image path
        """
        files_true = glob.glob('%s/%s/imgs/*/*'%(self.data_path, self.data_path))
        files_false = glob.glob('%s/%s_neg/imgs/*/*'%(self.data_path, self.data_path))
        n_true, n_false = len(files_true), len(files_false)
        n_train_true, n_dev_true = 0.9*n_true, 0.1*n_true
        n_train_false, n_dev_false = 0.9*n_false, 0.1*n_false
        train_true, dev_true = files_true[:n_train_true], files_true[n_train_true:]
        train_false, dev_false = files_false[:n_train_false], files_false[n_train_false:]
        self.train_files = train_true + train_false
        self.dev_files = dev_true + dev_false
        return self.train_files, self.dev_files

    def imgs_normalize(self,
                    im_array,
                    min=0, max=1):
        """Function to normalize images
        to scale the minimum pixel value to min
        and maximum pixel value to max
        :param im_array: Numpy array of images
                        to be normalized
        :param min: Minimum value of normalized images
        :param max: Maximum value of normalized images
        """
        old_min, old_max = self.px_min, self.px_max
        new_min, new_max = min, max
        im_array = (im_array-old_min)/(old_max-old_min)
        im_array = im_array*(new_max-new_min)
        im_array += new_min
        return im_array

    def compute_mean_std(self,write=False):
        """Function to compute the mean of
        all BSDS train images. Used for z-scoring
        the data.
        """
        mean_fn = join(self.tfr_path,'mean.npy')
        std_fn = join(self.tfr_path,'std.npy')
        all_imgs = []
        for im in self.im_fns:
            img = read_img(im)
            if img.max()>1.:
                img = self.imgs_normalize(img,min=0.,max=1.)
            if img.shape[0:2] != (321,481):
                img = img.transpose(1,0,2)
            all_imgs.append(img)
        all_imgs = np.array(all_imgs)
        self.mean_img = all_imgs.mean(axis=(0,1,2),keepdims=True)
        self.std_img = all_imgs.std(axis=(0,1,2),keepdims=True)
        if write:
            np.save(mean_fn, self.mean_img)
            np.save(std_fn, self.std_img)
        return self.mean_img, self.std_img

    def check_tfrecords(self, split='train'):
        """Function to check if a tfrecord file already
        exists to prevent overwriting.
        :param split: Train/val/test split of the tfrecord file
        """
        import glob
        tfr_fn = join(self.tfr_path,'%s.%s'%(split,
                                            self.tfr_ext
                                            ))
        exists = glob.glob(tfr_fn)
        return exists!=[]

    def write_tfrecords_split(self,split='train'):
        """Function to write tfrecords for BSDS.
            :param split: One of 'train' or 'val'
                           indicating the dataset
                           being written
        """
        import datetime
        def encode_shape(img_shape):
            # To encode shape in binary array for tfrecords
            feat_list = tf.train.Int64List(value = [img_shape[0],img_shape[1]])
            feat = tf.train.Feature(int64_list = feat_list)
            return feat

        def encode_img(image_data):
            # To encode image in a serialized string
            image_str = tf.compat.as_bytes(image_data.tostring())
            feat_list = tf.train.BytesList(value = [image_str])
            feat = tf.train.Feature(bytes_list = feat_list)
            return feat

        def encode_int(label_data):
            # To encode label in tfrecords
            feat_list = tf.train.Int64List(value=[label_data])
            feat = tf.train.Feature(int64_list=feat_list)
            return feat

        def encode_string(string_data):
            # To encode filename in tfrecords
            feat_list = tf.train.BytesList(value=[string_data])
            feat = tf.train.Feature(bytes_list=feat_list)
            return feat

        def convert_image(img_path):
            image_data = read_img(img_path)
            import ipdb; ipdb.set_trace()
            image_id = img_path.split('-')[-2] #.split('_')[0]
            image_id = int(image_id)
            label = 0
            if 'neg' not in img_path:
                label = 1
            # Store shape of image for reconstruction purposes
            image_shape = image_data.shape
            feat_dict = {
                        '%s/shape'%(split): encode_shape(image_shape),
                        '%s/image'%(split): encode_img(image_data),
                        '%s/label'%(split): encode_int(label),
                        '%s/file_id'%(split): encode_int(image_id)
                        }
            features = tf.train.Features(feature=feat_dict)
            example = tf.train.Example(features = features)
            return example
        # Making sure directory to save the tfrecord exists
        mkdir(self.tfr_path)
        timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ','-').replace(':','-').replace('.','-')
        if 'train' in split:
            im_fns = self.train_files
        else:
            im_fns = self.dev_files
        tfr_fn = join(self.tfr_path, '%s_%s_%s.%s' % (split, timestamp,
                                                len(im_fns),
                                                self.tfr_ext
                                                ))
        ########## Start writing tfrecords ##########
        # check for shuffle
        if self.shuffle:
            np.random.shuffle(im_fns)
        with tf.python_io.TFRecordWriter(tfr_fn) as writer:
            for im_fn in tqdm(im_fns,
                            desc='Writing tfrecords for %s'%(split)):
                example = convert_image(im_fn)
                if example is not None:
                    writer.write(example.SerializeToString())
                else:
                    print "Example is None"
        writer.close()

    def write_tfrecords(self):
        if self.write_train:
            self.write_tfrecords_split('train')
        if self.write_dev:
            self.write_tfrecords_split('dev')

    def get_tfr_iterator(self,
                        split,
                        batch_size=64,
                        ):
        """Function to read tfrecords for BSDS
        :param split: One of 'train' or 'val'
                       indicating the dataset
                       being written
        :param tfr_fn: Name of tfrecord file
                        being read"""
        import os
        # tfr_fn = join(self.tfr_path, '%s.%s'%(split,
        #                                     self.tfr_ext))
        tfr_fn = glob.glob('%s/%s*.%s'%(self.tfr_path, split, self.tfr_ext))
        tfr_fn.sort(key=os.path.getmtime)
        tfr_fn = tfr_fn[-1]
        print 'Loading %s'%(tfr_fn)

        def decode_feats(tfrecord):
            # Function to decode features written in
            # tfrecords using feature dictionary
            feat_dict = {
                        '%s/shape'%(split): tf.FixedLenFeature(
                                                [2], tf.int64),
                        '%s/image'%(split): tf.FixedLenFeature(
                                                [], tf.string),
                        '%s/label'%(split): tf.FixedLenFeature(
                                                [], tf.int64),
                        '%s/file_id'%(split): tf.FixedLenFeature(
                                                [], tf.int64),
                        }
            sample = tf.parse_single_example(tfrecord, feat_dict)
            # decoding shape
            shape = sample['%s/shape'%(split)]
            # decoding image
            img = tf.decode_raw(sample['%s/image'%(split)],tf.uint8)
            img = tf.reshape(img,[300,300,3])
            # decoding label (class indicator)
            label = sample['%s/label'%(split)]
            file_id = sample['%s/file_id'%(split)]
            return [img,shape,label,file_id]

        dataset = tf.data.TFRecordDataset([tfr_fn])
        dataset = dataset.map(decode_feats)
        dataset = dataset.batch(batch_size)
        print 'Batching at %s'%(batch_size)
        iterator = dataset.make_initializable_iterator()
        im_s_l = iterator.get_next()
        return im_s_l, iterator


def write_tfrecords(datapath):
    """
    Driver function to write tfrecords for open close dataset
    :return:
    """
    writer = pathfinder(
           data_path=datapath,
           img_shape_xy=(300,300),
           tfr_path='tfrecords', shuffle=True,
           im_ext='png',tfr_ext='tfrecord',
           )
    writer.write_tfrecords()
    return writer


def main():
    writer = write_tfrecords(sys.argv[1])

if __name__=='__main__':
    main()
