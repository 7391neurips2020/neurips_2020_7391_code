import numpy as np
import cv2
import tensorflow as tf
from os.path import join
import scipy.io as sio
from utils.file_utils import *
import glob
import sys
from tqdm import tqdm
from dataloaders.openclose_dataloader import openclose_dataloader
class openclose:
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
            write_test=True,
            ):
        self.data_path = data_path
        # Checking the sync
        self.tfr_path = join(data_path,'openclose_%s'%(tfr_path))
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
        self.write_test = write_test
        self.train_files, self.dev_files = self.create_train_test()


    def create_train_test(self):
        """Function to return image
        names from the given image path
        """
        files_close = glob.glob('%s/*closed_contours*/*png'%(self.data_path))
        files_open = glob.glob('%s/*open_contours*/*png'%(self.data_path))
        n_close, n_open = len(files_close), len(files_open)
        n_train_close, n_dev_close = 0.9*n_close, 0.1*n_close
        n_train_open, n_dev_open = 0.9*n_open, 0.1*n_open
        train_close, dev_close = files_close[:n_train_close], files_close[n_train_close:]
        train_open, dev_false = files_open[:n_train_open], files_open[n_train_open:]
        self.train_files = train_close + train_open
        self.dev_files = dev_close + dev_false
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
            image_id = img_path.split('-')[-2]
            image_id = int(image_id)
            # image_data = self.imgs_normalize(image_data)
            label = 0
            if 'closed' in img_path:
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
        if 'train' in split:
            im_fns = self.train_files
        else:
            im_fns = self.dev_files
        timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ','-').replace(':','-').replace('.','-')
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
        if self.write_test:
            self.write_tfrecords_split('test')
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
            img = tf.reshape(img,[256,256,3])
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
    writer = openclose(
           data_path=datapath,
           img_shape_xy=(256,256),
           tfr_path='tfrecords', shuffle=True,
           write_test=False, 
           im_ext='png',tfr_ext='tfrecord',
           )
    writer.write_tfrecords()
    return writer


def main():
    writer = write_tfrecords(sys.argv[1])

if __name__=='__main__':
    main()
