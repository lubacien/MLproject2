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
from U_net import *
from losses import *
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datagen import *

## Seeding
seed = 2
random.seed = seed
np.random.seed = seed
tf.seed = seed
epochs = 5

dataset_path = "../data/"
train_path = os.path.join(dataset_path, "training/")
#test_path = os.path.join(dataset_path, "test/")

imgs = []
labels=[]
imgpath = train_path + 'images/'
labelpath = train_path + 'groundtruth/'
files = os.listdir(imgpath)
for name in files:
    imgs.append(np.asarray(cv2.imread(imgpath+name)))
    labels.append(cv2.imread(labelpath+name))

image_size = 400
batch_size = 16
val_data_size = 10
#shuffle first
valid_imgs = imgs[:val_data_size]
train_imgs = imgs[val_data_size:]
valid_labels=labels[:val_data_size]
train_labels=labels[val_data_size:]

model = ResUNet(400)

model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_loss])

model.summary()

save_model_path = '/tmp/weights.h5'
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
'''
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

'''

history = model.fit(train_imgs, train_labels,
                   steps_per_epoch=int(np.ceil(len(imgs) / float(batch_size))),
                   epochs=epochs,
                   #validation_data=valid_imgs,
                   #validation_steps=int(np.ceil(len(imgs) / float(batch_size))),
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