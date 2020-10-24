#-*- coding: utf-8 -*-
import keras
import tensorflow

import keras.backend as K
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

import datetime
from PIL import Image

from config import config_def
import config_env

from dataset_loader import DatasetLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

config_dataset = config_env.config_dataset
config_model = config_env.config_model

batch_size = config_env.batch_size
epochs = config_env.epochs
verbose = config_env.verbose

img_dir = config_dataset.img_dir
image_size = config_dataset.image_size

if config_dataset.class_mode in ['categorical', 'binary']:
  create_model_func = lambda *args: config_model.create_model(config_dataset=config_dataset, *args)
else:
  create_model_func = config_model.create_model

datasetLoader = DatasetLoader(config_env, config_dataset, config_model, augumentation=True)

# Output filename
datetimePrefix = datetime.datetime.today().strftime("%Y%m%d_%H%M%S") + '_'
datasetPrefix = "%s_%s" % (config_model.modelName, config_dataset.datasetName)
filePrefix = datetimePrefix + datasetPrefix

model_save_name = '../h5files/' + filePrefix +'.{acc:.2f}-{epoch:02d}.hdf5'
os.makedirs('../log/', exist_ok = True)
csvlog_name = '../log/' + filePrefix + '.log'

resultsDir = '../testResults'

# ModelCheckpoint
weights_dir = '../weights/'
if os.path.exists(weights_dir) == False:
  os.mkdir(weights_dir)
monitor = config_model.monitorKey
model_checkpoint = ModelCheckpoint(
    os.path.join(weights_dir, filePrefix + "_" + monitor + "{" + monitor + ":.3f}.hdf5"),
    monitor = config_model.monitorKey,
    verbose = 1 if verbose else 0,
    save_best_only = True,
    save_weights_only = True,
    period = 1
)
bestWeightsFile = os.path.join(weights_dir, datasetPrefix + "_best.hdf5")
model_checkpoint_best = ModelCheckpoint(
    bestWeightsFile,
    monitor = config_model.monitorKey,
    verbose = 0,
    save_best_only = True,
    save_weights_only = True,
    period = 1
)

predict_only_model_filepath = bestWeightsFile

# log for TensorBoard

if config_env.tflog_dir:
  os.makedirs(config_env.tflog_dir, exist_ok=True)
  tensorBoard = TensorBoard(log_dir = config_env.tflog_dir)

K.clear_session()

img_width=image_size[0]
img_height=image_size[1]

# Initialize Tensorflow session.
tf_config = tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(allow_growth=True))
session = tensorflow.Session(config=tf_config)
K.tensorflow_backend.set_session(session)


csv_logger = CSVLogger(csvlog_name)

script_path = os.path.abspath(__file__)


model = create_model_func()
model.summary()
model_name = model_save_name
history = model.fit_generator(
 datasetLoader.flow(config_def.TRAIN),
 epochs = epochs,
 validation_data = datasetLoader.flow(config_def.VALIDATION),
 verbose=1,
 callbacks = config_model.tfCallbacks(verbose) + [
   csv_logger,
   model_checkpoint,
   model_checkpoint_best,
   tensorBoard]
)

