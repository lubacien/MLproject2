

import matplotlib.pyplot as plt

from PIL import Image
from sklearn import linear_model
from helpers import *
from preprocessing import *
from mask_to_submission import *
from submission_to_mask import *
from imaging import *
import numpy as np
import matplotlib.image as mpimg


n = 100 # number of images to be loaded

imgs, gt_imgs = load_images(n)
print(np.array(imgs).shape)

# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels

img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

print(np.array(img_patches).shape)

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

print(img_patches.shape)

#Extracting features
X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(len(set(Y)) ))


Y0 = [i for i, j in enumerate(Y) if j == 0]
Y1 = [i for i, j in enumerate(Y) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')


#training the model
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
logreg.fit(X, Y)

#creating submission images

for i in range(1, 51):
    image_filename = '../data/test_set_images/test_' + str(i) + '/test_'+ str(i) +'.png'

    test=extract_img_features(image_filename,patch_size)
    pred=logreg.predict(test)
    #pred[np.where(pred == 0)] = 0
    #pred[np.where(pred == 1)] = 254
    image = load_image(image_filename)
    w = image.shape[0]
    h = image.shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, pred)
    predicted_im = binary_to_uint8(predicted_im)
    Image.fromarray(predicted_im).save('../data/submission/prediction' + '%.3d' % i + '.png')
    #mpimg.imsave('../data/submission/predictionim' + '%.3d' % i + '.png', predicted_im,vmin=0,vmax=255)

#creating submission file
submission_filename = '../data/submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = '../data/submission/predictionim' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)

masks_to_submission(submission_filename, *image_filenames)
