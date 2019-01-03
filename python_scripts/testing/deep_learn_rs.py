from __future__ import print_function, division
import numpy as np
import random
import os
import glob
import datetime
import pandas as pd
import time
import h5py
import csv
import re
from PIL import Image as pil_image
from sklearn.preprocessing import LabelEncoder
import argparse
from scipy.misc import imresize, imsave
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.utils import shuffle
from PIL import Image, ImageChops, ImageOps
import sys
#from keras.utils.training_utils import multi_gpu_model
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-g", "--gpus", type=int, default=1,
#	help="# of GPUs to use for training")
#args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G =1 #args["gpus"]
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gzip
import pickle
from collections import Counter
import keras
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, model_from_json
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score
from tqdm import tqdm as tqdm

from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Activation, Dropout, Flatten, Dense


cwd = os.getcwd()

def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img

def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def list_pictures(directory, ext='jpg|jpeg|bmp|png|tif'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

def database_image(shape=(256,256,3),
                   directory="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/UCMerced_LandUse/full/"):
    images=list_pictures(directory)
    X,y,y_name=[],[],[]
    for im in images:
        im_print=load_img(im)
        im_array=img_to_array(im_print)
        if im_array.shape[0]>shape[0]:
            result=im_array[:shape[0],:shape[1],:shape[2]]
        else:
            result = np.zeros(shape)
            result[:im_array.shape[0],:im_array.shape[1],:im_array.shape[2]] = im_array
        X.append(result)
        y.append(im.split("/")[-1][:-6])
        y_name.append(im)
    return X,y,y_name

def fbs(y_true, y_pred, threshold_shift=0., beta=1):

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

def f2_score(y_true, y_pred):
    # fbs throws a confusing error if inputs are not numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbs(y_true, y_pred, beta=2, average='samples')

def instantiate(n_classes, n_dense=1024, resnet_json="resnet50_mod.json", target_size=(256,256,3), verbose=1):
    """
    Instantiate the resnet 50.
    """
    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=target_size)
    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # and a final logistic layer
    predictions = Dense(n_classes, activation='sigmoid')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=[fbs])
    # serialize model to json
    model_json = model.to_json()
    with open(resnet_json, "w") as iOF:
        iOF.write(model_json)
    return base_model, model

def finetune(base_model, model, X_train, y_train, X_val, y_val,
             epochs_1=1000, patience_1=2,
             patience_lr=1, batch_size=32,
             nb_train_samples=1600, nb_validation_samples=500,
             img_width=256, img_height=256, class_imbalance=False,
             resnet_h5_1="resnet50_fine_tuned_1.h5",
             resnet_h5_check_point_1="resnet50_fine_tuned_check_point_1.h5",
             layer_names_file="resnet50_mod_layer_names.txt", verbose=1):
    """
    Finetune the resnet 50.
    """
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    with open(layer_names_file, "w") as iOF:
        for ix, layer in enumerate(model.layers):
            iOF.write("%d, %s\n"%(ix, layer.name))
            if verbose >= 4: print(ix, layer.name)
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        fill_mode='reflect')
    train_datagen.fit(X_train)
    # this is the augmentation configuration we will use for testing:
    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        fill_mode='reflect')
    test_datagen.fit(X_train)
    # define train & val data generators
    train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size,shuffle=True)
    validation_generator = test_datagen.flow(X_val,y_val,batch_size=batch_size,shuffle=True)
    # get class weights
    if class_imbalance:
        class_weight = get_class_weights(np.sum(y_train, axis=0), smooth_factor=0.1)
    else:
        class_weight = None
    # train the model on the new data for a few epochs on the batches generated by datagen.flow().
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience_1),
                   ModelCheckpoint(filepath=resnet_h5_check_point_1, save_best_only=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr)],
        class_weight=class_weight)
    # save weights just in case
    model.save_weights(resnet_h5_1)


X,encoded,encoded_name=database_image()
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(encoded)
y = to_categorical(integer_encoded)
dic_int_to_label={}
for lab,nb in zip(encoded,integer_encoded):
    dic_int_to_label.setdefault(nb,lab)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=4)
_,_,name_train,name_test=train_test_split(X,encoded_name,test_size=0.2, random_state=4)
X_train=np.stack(X_train)
X_test=np.stack(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

verbose=1
model_dir="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/UCMerced_LandUse/"
if verbose >= 1:
    print("\tInstantiating ResNet50 (fold )...")
n_classes = y_train.shape[1]
base_model, model = instantiate(n_classes, n_dense=1024, resnet_json=model_dir+"resnet50_mod_.json", target_size=(256,256,3), verbose=verbose)
#model = multi_gpu_model(model, gpus=G)

if verbose >= 1: print("\tFine-tuning ResNet50 first pass (fold )...")
finetune(base_model, model, X_train, y_train, X_test, y_test, batch_size=20, epochs_1=200,
         nb_train_samples=len(y_train), nb_validation_samples=len(y_test),
         img_width=256, img_height=256,
         patience_1=100, patience_lr=100, class_imbalance=False,
         resnet_h5_1=model_dir+"resnet50_fine_tuned_1_.h5",
         resnet_h5_check_point_1=model_dir+"resnet50_fine_tuned_check_point_1_.h5",
         layer_names_file=model_dir+"resnet50_mod_layer_names.txt",
         verbose=verbose)

