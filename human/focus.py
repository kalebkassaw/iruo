import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

OVIS_PATH = ""
PATH_TO_IRUO = ""

floc = lambda fname: f'{PATH_TO_IRUO}/images/%s' % fname
mloc = lambda fname: f'{PATH_TO_IRUO}/seg-masks/%s' % fname
annos = lambda occ_level: pd.read_csv(f'{PATH_TO_IRUO}/annos/val%i.txt' % occ_level, sep=' ', header=0, names=['fname', 'cls'])
px = lambda img: int(np.prod(img.shape) / img.shape[-1])
blur = lambda img: cv2.Laplacian(img, cv2.CV_64F).var()

def focus(ann):
    pxs, blurs = [], []
    for f in tqdm(ann.fname.iloc, desc='Running focus calculator', total=len(ann)):
        im = cv2.imread(floc(f))
        pxs.append(px(im))
        blurs.append(blur(im))
        del im
    ann['px'] = pxs
    ann['blur'] = blurs

def filter_clear(ann):
    filter_px = ann[ann.px > 10000]
    return filter_px[filter_px.blur > 20]