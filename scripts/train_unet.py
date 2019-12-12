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
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from U_net import *
from losses import *

import os


img_shape = (400, 400, 3)
batch_size = 3
epochs = 5
num_images=100

def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair
  img_str = tf.read_file(fname)
  img = tf.image.decode_png(img_str, channels=3)

  label_img_str = tf.read_file(label_path)
  label_img = tf.image.decode_png(label_img_str,channels=1)
  print(label_img.shape)
  print(np.unique(label_img))
  # The label image should only have values of 1 or 0, indicating pixel wise
  # object (car) or not (background). We take the first channel only.
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

model = make_model()

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

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()