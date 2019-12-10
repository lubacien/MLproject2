from tensorflow.python.keras import models
import matplotlib.image as mpimg
import os
from PIL import Image
import sys
from mask_to_submission import *
import tensorflow as tf
import numpy as np

PIXEL_DEPTH=255

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

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

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def getprediction(img):
    #takes as input a 608x608 image and gives a 608x608 prediction based on 400x400 predictions from network
    subimgs = np.empty((4,400,400,3))

    #we create 4 different predictions

    subimgs[0]=img[0:400,0:400,:]
    subimgs[1]=img[208:608,0:400,:]
    subimgs[2]=img[0:400,208:608,:]
    subimgs[3]=img[208:608,208:608,:]
    print(np.asarray(subimgs).shape)
    preds=model.predict(subimgs)

    output=np.empty((608,608))
    #we first fill the 4 corners with the only predictions for the corners

    output[0:208,0:208]=preds[0,0:208,0:208,0]
    output[400:608, 0:208] = preds[1,192:400, 0:208, 0]
    output[0:208, 400:608] = preds[2,0:208, 192:400, 0]
    output[400:608, 400:608] = preds[3,192:400, 192:400, 0]
    #then we fill the 4 middle rectangles with means of 2 predictions
    recttop=np.empty((2,192,208))
    rectbottom = np.empty((2, 192, 208))
    rectleft = np.empty((2, 208, 192))
    rectright = np.empty((2, 208, 192))

    recttop[0]=preds[0,208:400,0:208,0]
    recttop[1] = preds[1,0:192,0:208, 0]
    output[208:400,0:208]=np.mean(recttop,axis=0)

    rectbottom[0] = preds[2, 208:400, 0:208, 0]
    rectbottom[1] = preds[3, 0:192, 0:208, 0]
    output[208:400, 400:608] = np.mean(rectbottom, axis=0)

    rectleft[0] = preds[0, 0:208, 208:400, 0]
    rectleft[1] = preds[2, 0:208, 0:192, 0]
    output[0:208, 208:400] = np.mean(rectleft, axis=0)

    rectright[0] = preds[1, 192:400, 208:400, 0]
    rectright[1] = preds[3, 192:400, 0:192, 0]
    output[400:608, 208:400] = np.mean(rectright, axis=0)

    #then we fill the center with means of all predictions
    tomean=np.empty((4,192,192))
    tomean[0]=preds[0,208:400,208:400,0]
    tomean[1]=preds[1,0:192,208:400,0]
    tomean[2] = preds[2, 208:400, 0:192, 0]
    tomean[3] = preds[3, 0:192, 0:192, 0]
    output[208:400,208:400]=np.mean(tomean,axis=0)

    return output

save_model_path = '../results/weights_5epochs.hdf5'
# Alternatively, load the weights directly: model.load_weights(save_model_path)
model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                          'dice_loss': dice_loss})
print("Running prediction on test set")
prediction_training_dir = "../data/submissionimages/"
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)


for i in range(1, 50 + 1):
    test_data_filename=("../data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png")
    img = mpimg.imread(test_data_filename)
    pimg = getprediction(img)
    pimg = img_float_to_uint8(pimg)
    Image.fromarray(pimg).save(prediction_training_dir + 'prediction' + '%.3d' % i + '.png')

'''
for i in range(50):
    imgs[i]
'''



# creating submission file
submission_filename = '../data/submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = '../data/submissionimages/prediction' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)