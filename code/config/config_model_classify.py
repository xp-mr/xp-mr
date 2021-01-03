#*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import keras
import tensorflow
import os

#modelName = 'Xception'
#modelName = 'InceptionV3'
modelName = 'ResNet50'
#modelName = 'DenseNet121'
if modelName == 'ResNet50':
  from keras.applications.resnet50 import ResNet50 as ModelFactory, preprocess_input, decode_predictions
  gradcam_layer_name='activation_49'
elif modelName == 'VGG16':
  from keras.applications.vgg16 import VGG16 as ModelFactory, preprocess_input, decode_predictions
  gradcam_layer_name='block5_conv3'
elif modelName == 'Xception':
  from keras.applications.xception import Xception as ModelFactory, preprocess_input, decode_predictions
  gradcam_layer_name='block14_sepconv2_act'
elif modelName == 'DenseNet121':
  from keras.applications.densenet import DenseNet121 as ModelFactory, preprocess_input, decode_predictions
  gradcam_layer_name=None
elif modelName == 'InceptionV3':
  from keras.applications.inception_v3 import InceptionV3 as ModelFactory, preprocess_input, decode_predictions
  gradcam_layer_name='mixed10' 
else:
  raise Exception("Unknown classname `{}`".format(modelName))

from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.callbacks import CSVLogger, Callback
from keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from . import config_def

from keras import backend as K

monitorKey = 'val_loss'

def create_model(config_dataset):

  base_model = ModelFactory(
    include_top = False,
    weights = None,
    #weights = "imagenet",
    input_shape = None
  )
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  if config_dataset.objectiveType == config_def.OBJECTIVE_TYPE_CATEGORICAL:
    num_classes = 2 # edit if needed
    x = Dense(1024, activation='sigmoid')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    lossFunction = 'categorical_crossentropy'
  elif config_dataset.objectiveType == config_def.OBJECTIVE_TYPE_BINARY:
    predictions = Dense(1, activation='sigmoid')(x)
    lossFunction = 'binary_crossentropy'
  else: # REGRESSION
    predictions = Dense(1, activation='sigmoid')(x)
    lossFunction = 'mse'
  
  model = Model(inputs = base_model.input, outputs = predictions)
  
  model.compile(
    optimizer = Adam(),
    loss = lossFunction,
    metrics = ['accuracy','mse','mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error', 'cosine_proximity', 'msle'],
  )

  return model


def tfCallbacks(verbose = True):
  nVerbose = 1 if verbose else 0
  return [
    # reduce learning rate
    ReduceLROnPlateau(
        monitor = monitorKey,
        factor = 0.1,
        patience = 3,
        verbose = nVerbose
    ),
    # EarlyStopping
    EarlyStopping(
        monitor = monitorKey,
        patience = 100,
        verbose = nVerbose
    )
  ]

