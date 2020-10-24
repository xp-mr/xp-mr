#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import keras
#import tensorflow
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from . import config_def
import statistics

color_mode = 'rgb'
dataframe_x_col = 'png'
dataframe_y_col = 'class'
dataframe_y_src_col = 'value'
file_ext = 'png'

image_size=[300, 300]

datasetName = 'Xp2EF-PACS-MR-Class'
base_dir = '/path/to/dataset/' # TODO
img_dir = os.path.join(base_dir, 'imgdir') # TODO
dataset_file = os.path.join(base_dir, 'dataset.csv') # TODO

classes = ['normal', 'abnormal']

objectiveType = config_def.OBJECTIVE_TYPE_BINARY


if objectiveType == config_def.OBJECTIVE_TYPE_BINARY:
  # append class index as prefix to avoid lexicographic sorting in BINARY.
  # https://github.com/keras-team/keras-preprocessing/issues/289
  classes = list ( map(lambda x: "{:0>3}_{}".format(x[0], x[1]), enumerate(classes)) )

# ----
num_classes = len(classes)

if objectiveType == config_def.OBJECTIVE_TYPE_CATEGORICAL:
  classify = lambda x: x
  class_mode = 'categorical'
elif objectiveType == config_def.OBJECTIVE_TYPE_BINARY:
  classify = lambda x: classes[x]
  class_mode = 'binary'
else: # regression
  classify = lambda x: x
  class_mode = 'other'

def loadDataframe(subset, useDownSample=True, useUpSample=False):
  config_def.verifySubset(subset)
  print("config_dataset.loadDataframe(): load %s from %s" % (subset, dataset_file))
  df = pd.read_csv(dataset_file)
  df = df[df['split'] == subset].reset_index()[[
    dataframe_x_col, dataframe_y_src_col]]
  df[dataframe_y_col] = df[dataframe_y_src_col].map(lambda x: classify(x))
  res = df
  if objectiveType == config_def.OBJECTIVE_TYPE_CATEGORICAL:
    res[dataframe_y_col] = res[dataframe_y_col].map(lambda x: [x])
  #elif objectiveType == config_def.OBJECTIVE_TYPE_BINARY:
  #  # noop
  #else: # Regression
  #  # noop
  return res

def createGenerator(preprocess_input=None, augumentation=False):
    if not augumentation:
      return ImageDataGenerator(
          preprocessing_function=preprocess_input
      )
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        #featurewise_center=False,
        #samplewise_center=False,
        #featurewise_std_normalization=False,
        #samplewise_std_normalization=False,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        #brightness_range= [0.9,1.0],
        #horizontal_flip = False,
        #vertical_flip = False,
        #zoom_range = 0.1,
        #zoom_range = False,
        #channel_shift_range=30,
        #fill_mode='reflect'
        fill_mode='constant', cval=0
    )
