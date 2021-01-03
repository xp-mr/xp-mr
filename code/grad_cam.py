import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.python.framework import ops
import datetime
from config import config_def
import config_env
from dataset_loader import DatasetLoader

config_dataset = config_env.config_dataset
config_model = config_env.config_model

preprocess_input = config_model.preprocess_input
decode_predictions = config_model.decode_predictions

fileSuffix = '_gradcam'
datetimePrefix = datetime.datetime.today().strftime("%Y%m%d_%H%M%S") + '_'
datasetPrefix = "%s_%s" % (config_model.modelName, config_dataset.datasetName)
filePrefix = datetimePrefix + datasetPrefix
weights_dir = '../weights/'
bestWeightsFile = os.path.join(weights_dir, datasetPrefix + "_best.hdf5")

testOutputDir = os.path.join('../testResults', filePrefix + fileSuffix)
os.makedirs(testOutputDir, exist_ok=True)
def formatPredictResultFilename(index, expected, actual, filename):
  if config_dataset.class_mode == 'categorical':
    classes = config_dataset.classes
    expected = classes[expected]
    actual = classes[actual] if actual != None else None
    fmt='{}'
  else:
    fmt='{:.2f}'
  if actual == None: fmt = '{}'

  return ('{:0>8}_expect_{}_pred_' + fmt + '_{}').format(index, expected, actual, os.path.basename(filename))

datasetLoader = DatasetLoader(config_env, config_dataset, config_model, augumentation=False)
weightsFile = bestWeightsFile
cachedModel = None
def build_model(resetCache = False):
    global cachedModel
    if not resetCache and cachedModel: return cachedModel
    model = config_model.create_model(config_dataset)
    print("load weights: {}".format(weightsFile))
    model.load_weights(weightsFile)
    #model.summary()
    cachedModel = model
    return model

# H, W = 224, 224 # Input shape, defined by the model (model.input_shape)
(W, H) = config_dataset.image_size
# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x

def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    return cam
    
def compute_saliency(model, img_path, realValue, layer_name, index=None):
    preprocessed_input = load_image(img_path)

    ## Removed prediction to avoid memory leaks.
    # predictions = model.predict(preprocessed_input)
    predictions = None

    actual = predictions
    expected = realValue

    if config_dataset.class_mode == 'categorical':
      expected = realValue[0] # [class]. [0] [3] [5] 
      actual = np.argmax(actual) if actual else None # one hot.
      cls = expected
    elif config_dataset.class_mode == 'binary':
      classes = config_dataset.classes
      expected = classes.index(expected) # '000_normal'
      actual = actual[0][0] if actual else None
      cls = 0
    else: # regression
      actual = actual[0][0] if actual else None
      cls = 0

    #class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    #print("Explanation for '{}'".format(class_name))
    
    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)

    jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
    prefix = formatPredictResultFilename(index, expected, actual, img_path)
    cv2.imwrite(os.path.join(testOutputDir, prefix + '_gradcam.jpg'), np.uint8(jetcam))
    
    return gradcam

def printInfo(s):
  print('{:%Y-%m-%d %H:%M:%S}: {}'.format(datetime.datetime.now(), s))

layer_name = config_model.gradcam_layer_name
if not layer_name:
  print("Missing variable gradcam_layer_name in config_model.")
  exit(1);
if __name__ == '__main__':
    printInfo("output dir={}".format(testOutputDir))
    modelReuseCount = 0
    for index, (filepath, value) in enumerate(datasetLoader.dictFiles('test').items()):
      printInfo(filepath)
      # Reset Tensorflow session periodically to avoid memory leaks.
      if modelReuseCount == 0:
        printInfo("Reset Tensorflow session.")
        K.clear_session()
        model = build_model(resetCache = True)
        printInfo("... done")
      modelReuseCount = (modelReuseCount + 1) % 10
      gradcam = compute_saliency(model, layer_name=layer_name,
                                 img_path=filepath, realValue=value, index=index)

