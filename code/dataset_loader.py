#-*- coding: utf-8 -*-
import os
from config import config_def
import numpy as np
import pandas as pd

class DatasetLoader:
  generator = None
  config_env = None
  config_dataset = None
  config_model = None
  shuffle = True

  def __init__(self, config_env, config_dataset, config_model, augumentation):
    self.config_env = config_env
    self.config_dataset = config_dataset
    self.config_model = config_model
    self.generator = self.config_dataset.createGenerator(
      preprocess_input=self.config_model.preprocess_input,
      augumentation=augumentation)

  '''
    subset: TRAIN, VALIDATION, or TEST that are defined in config_def
  '''
  def flow(self, subset, with_filepaths=False):
    config_def.verifySubset(subset)
    dataset = self.config_dataset
    env = self.config_env

    if dataset.objectiveType == config_def.OBJECTIVE_TYPE_BINARY:
      classes = dataset.classes
    else:
      classes = None
    # require keras >=2.2.4
    # https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model
    res = self.generator.flow_from_dataframe(
      dataframe = self.loadDataframe(subset),
      directory = dataset.img_dir,
      x_col = dataset.dataframe_x_col,
      y_col = dataset.dataframe_y_col,
      target_size= (dataset.image_size[1], dataset.image_size[0],), # height width tuple
      color_mode = dataset.color_mode,
      class_mode= dataset.class_mode,
      classes = classes,
      batch_size= env.batch_size,
      shuffle = self.shuffle,
      save_format= dataset.file_ext
    )
    if with_filepaths:
      res = self._wrapFlowWithFile(res)
    return res

  def dictFiles(self, subset):
    config_def.verifySubset(subset)
    dataset = self.config_dataset
    env = self.config_env

    x_col = dataset.dataframe_x_col
    y_col = dataset.dataframe_y_col
    
    dataframe = self.loadDataframe(subset)
    directory = dataset.img_dir
    return {os.path.join(directory, row[x_col]):row[y_col] for i,row in dataframe.iterrows() }
  def loadDataframe(self, subset):
    dataset = self.config_dataset
    dataframe = dataset.loadDataframe(subset)
    return dataframe

  def _wrapFlowWithFile(self, flow):
    return FlowWithFile(flow)
    
class FlowWithFile:
    def __init__(self, datagen):
        self.datagen = datagen
        self.cnt = 0
        self.batches_per_epoch = datagen.samples // datagen.batch_size + (datagen.samples % datagen.batch_size > 0)

    def __iter__(self):
        return self
    def __len__(self):
        return len(self.datagen)

    def __next__(self):
        datagen = self.datagen
        batch = next(datagen)
        current_index = ((datagen.batch_index-1) * datagen.batch_size)
        if current_index < 0:
            if datagen.samples % datagen.batch_size > 0:
                current_index = max(0,datagen.samples - datagen.samples % datagen.batch_size)
            else:
                current_index = max(0,datagen.samples - datagen.batch_size)
        index_array = datagen.index_array[current_index:current_index + datagen.batch_size].tolist()
        img_paths = [datagen.filepaths[idx] for idx in index_array]
        self.cnt = (self.cnt + 1) % self.batches_per_epoch
        return batch + (img_paths,)
