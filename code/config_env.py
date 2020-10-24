#-*- coding: utf-8 -*-
import os
from config import config_dataset_classify as config_dataset
from config import config_model_classify as config_model

import datetime

datetimeString = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")

# tensorflow logging dir.
tflog_dir = os.path.join(os.path.dirname(__file__), '../tflog', datetimeString)
os.makedirs(tflog_dir, exist_ok=True)

batch_size=8

epochs=100

verbose = True


