#!/usr/bin/python
import glob
import numpy as np
import pandas as pd
import pickle
import json
# import cv2
# import matplotlib.pyplot as plt
import os

def parse_json(fn):
    # Function to parse json and
    # return parameter dictionary
    with open(fn) as f:
        config = json.load(f)
    return config

def load_pickle(fn):
    # Function to load a pickle
    # file and return the stored data
    pkl = pickle.load(open(fn,'r'))
    return pkl

def dump_pickle(fn,data):
    # Function to dump data to a pickle
    pkl = pickle.dump(open(fn,'w'),data)
    return pkl

def mkdir(dir_path):
    """Function to create new directory
    with absolute path dir_path"""
    if glob.glob(dir_path) == []:
        os.mkdir(dir_path)

def read_img(im_path):
    """Function to read and return
    the images at im_path"""
    im = cv2.imread(im_path)
    return im

def ret_datetime():
    import datetime
    date_time = str(datetime.datetime.now()).replace(' ','_') \
                                            .replace('-','_') \
                                            .split('.')[0] \
                                            .replace(':','_')
    return date_time
