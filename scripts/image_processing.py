import cv2 as cv
import numpy as np
import os
from mask_to_submission import *
from tqdm import tqdm

def open_image(im_in):
    
    im_th = cv.bitwise_not(im_in)
    #th, im_th = cv.threshold(im_in, 125, 255, cv.THRESH_BINARY_INV)
    DISC = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    im_disc = cv.morphologyEx(im_th, cv.MORPH_OPEN, DISC)
    im_out = cv.bitwise_not(im_disc)

    return im_out

def floodfill_image(im_in):
    im_inv = cv.bitwise_not(im_in)
    
    th, im_th = cv.threshold(im_inv, 250, 255, cv.THRESH_BINARY)

    im_floodfill = im_th.copy()

    h,w = im_inv.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)


    for row, r in zip(im_inv.T[:], range(h)):
        if row[0] == 0:
            cv.floodFill(im_floodfill, mask, (r, 0), 255)
        if row[h-1] == 0:
            cv.floodFill(im_floodfill, mask, (r, w-1), 255)       
            
            
    for col, c in zip(im_inv[:], range(w)):
        if col[0] == 0:
            cv.floodFill(im_floodfill, mask, (0, c), 255)
        if col[w-1] == 0:
            cv.floodFill(im_floodfill, mask, (w-1, c), 255)
                            
    im_floodfill_inv = cv.bitwise_not(im_floodfill)    

    final = im_floodfill_inv | cv.bitwise_not(im_in)

    im_out = cv.bitwise_not(final)

    return im_out


mod_dir = '../data/sub_mod/'
if not os.path.isdir(mod_dir):
    os.mkdir(mod_dir)

predictions = []
submission_filename = '../data/mod_sub.csv'

for i in tqdm(range(1,51)):
    image_filename = '../data/submissionimages/prediction' + '%.3d' % i + '.png'

    im_in = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)

    im_open = open_image(im_in)
    im_flood = floodfill_image(im_open)

    mod_path = mod_dir + 'modified_' + '%.3d' % i + '.png'
    cv.imwrite(mod_path, im_flood)

    predictions.append(mod_path)

masks_to_submission(submission_filename, *predictions)
    

