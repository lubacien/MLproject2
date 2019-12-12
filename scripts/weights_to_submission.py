from tensorflow.python.keras import models
import matplotlib.image as mpimg
import os
from PIL import Image
import sys
from mask_to_submission import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from U_net import *
from losses import *

PIXEL_DEPTH=255




def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def prediction_to_class(image):
    threshold = 0.5
    out=np.empty(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > threshold:
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out

def getprediction(img,model):
    #more logical but works less well than predict_test_img?

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

def predict_test_img(img, model):
    """
    Function that takes a 608x608 image and returns it's predicted goundtruth by predicting
    4 corners of 400x400 then assembling them.
    """
    a = img[:400, :400]
    b = img[:400, 208:]
    c = img[208:, 208:]
    d = img[208:, :400]
    pred = model.predict(np.array(([a, b, c, d])))
    e = np.zeros((608, 608,1))
    e[:400, :400] = pred[0]
    e[:400, 208:] = pred[1]
    e[208:, 208:] = pred[2]
    e[208:, :400] = pred[3]
    e=e.reshape((608,608))
    return e

save_model_path = '../results/weights30_epochs_singlelayer.hdf5'
model = make_model2()
model.load_weights(save_model_path)

#model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                   #                                       'dice_loss': dice_loss})

prediction_training_dir = "../data/submissionimages/"
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)


for i in range(1, 51):
    test_data_filename=("../data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png")

    img = mpimg.imread(test_data_filename) #gives outputs between 0 and 1
    img=img_float_to_uint8(img) #converts them to be read by the NN
    pimg = getprediction(img, model)
    print(np.unique(pimg))
    #pimg = prediction_to_class(pimg) #ca change presque rien de faire ou pas faire ca
    pimg = img_float_to_uint8(pimg)
    Image.fromarray(pimg).save(prediction_training_dir + 'prediction' + '%.3d' % i + '.png')


'''
for i in range(50):
    imgs[i]
'''

'''
x_test_filenames = []
for i in range(1, 50 + 1):
    test_data_filename=("../data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png")
    x_test_filenames.append(test_data_filename)

test_ds = get_baseline_dataset(x_test_filenames)

# Let's visualize some of the outputs
data_aug_iter = test_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()

# Running next element in our graph will produce a batch of images
plt.figure(figsize=(10, 20))
for i in range(5):
    batch_of_imgs = tf.keras.backend.get_session().run(next_element)
    img = batch_of_imgs[0]
    predicted_label = model.predict(batch_of_imgs)[0]

    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(img)
    plt.title("Input image")

    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(label[0, :, :, 0])
    plt.title("Actual Mask")
    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(predicted_label[:, :, 0])
    plt.title("Predicted Mask")
plt.suptitle("Examples of Input Image, Label, and Prediction")
plt.show()
'''
# creating submission file
submission_filename = '../data/submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = '../data/submissionimages/prediction' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)