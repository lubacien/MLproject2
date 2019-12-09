import os
import glob
import zipfile
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import os


img_shape = (400, 400, 3)
batch_size = 3
epochs = 5
num_images=10

def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair
  img_str = tf.read_file(fname)
  img = tf.image.decode_jpeg(img_str, channels=3)

  label_img_str = tf.read_file(label_path)
  # These are gif images so they return as (num_frames, h, w, c)
  label_img = tf.image.decode_gif(label_img_str)[0]
  # The label image should only have values of 1 or 0, indicating pixel wise
  # object (car) or not (background). We take the first channel only.
  label_img = label_img[:, :, 0]
  label_img = tf.expand_dims(label_img, axis=-1)
  return img, label_img

def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=None,
                         threads=5,
                         batch_size=batch_size,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    #if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
     #   assert batch_size == 1, "Batching images must be of the same size"

    #dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder

def encoder_block_5x5(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((5, 5), strides=(5, 5))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def decoder_block_5x5(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (5, 5), strides=(5, 5), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

inputs = layers.Input(shape=img_shape)
# 400
print(inputs.shape)
encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 200
print(encoder0_pool.shape)
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 100
print(encoder1_pool.shape)
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 50
print(encoder2_pool.shape)
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 25
print(encoder3_pool.shape)
encoder4_pool, encoder4 = encoder_block_5x5(encoder3_pool, 512)
# 5
print(encoder4_pool.shape)
center = conv_block(encoder4_pool, 1024)
# center
print(center.shape)
decoder4 = decoder_block_5x5(center, encoder4, 512)
# 25
print(decoder4.shape)
decoder3 = decoder_block(decoder4, encoder3, 256)
# 32
print(decoder3.shape)
decoder2 = decoder_block(decoder3, encoder2, 128)
# 64
print(decoder2.shape)
decoder1 = decoder_block(decoder2, encoder1, 64)
# 128
print(decoder1.shape)
decoder0 = decoder_block(decoder1, encoder0, 32)
# 256
print(decoder0.shape)
outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
print(outputs.shape)

model = models.Model(inputs=[inputs], outputs=[outputs])

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_loss])

model.summary()

save_model_path = '/tmp/weights.hdf5'
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)

#Loading the dataset:
img_dir='../data/training/images/'
label_dir='../data/training/groundtruth/'
x_train_filenames = []
y_train_filenames = []

for img_id in range(1,num_images+1):
  x_train_filenames.append(os.path.join(img_dir, "satImage_%.3d" % img_id + ".png"))
  y_train_filenames.append(os.path.join(label_dir, "satImage_%.3d" % img_id + ".png"))

x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
                    train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=42)

train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames,
                                batch_size=batch_size)
val_ds = get_baseline_dataset(x_val_filenames,
                                y_val_filenames,
                                batch_size=batch_size)


history = model.fit(train_ds,
                   steps_per_epoch=int(np.ceil(num_images / float(batch_size))),
                   epochs=epochs,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(num_images / float(batch_size))),
                   callbacks=[cp])