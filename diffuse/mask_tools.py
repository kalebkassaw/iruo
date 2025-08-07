import numpy as np
import pandas as pd
import cv2
import os

OVIS_PATH = ""
PATH_TO_IRUO = ""

def _get_mask(img_name, mask_dir=f'{PATH_TO_IRUO}/seg-masks'):
    img = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, [224, 224])
    return img

def _get_image(img_name, img_dir=f'{PATH_TO_IRUO}/images'):
    img = cv2.imread(os.path.join(img_dir, img_name))
    img = cv2.resize(img, [224, 224])
    return img

def is_in_mask(xcoord, ycoord, img_name):
    try:
        img = _get_mask(img_name)
        # convert point to grid
        xcoord = int(xcoord * img.shape[1])
        ycoord = int(ycoord * img.shape[0])
        return img[ycoord, xcoord] > 128
    except:
        return pd.NA