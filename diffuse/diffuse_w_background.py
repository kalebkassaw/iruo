import numpy as np
import pandas as pd
import cv2
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt
import os
from class_tools import *
from mask_tools import is_in_mask
import json
from mask_tools import _get_image as get_img
from mask_tools import _get_mask as get_mask
from pptx import Presentation
from pptx.util import Inches
from scipy import ndimage

OVIS_PATH = ""
PATH_TO_IRUO = ""

img_dir = f'{PATH_TO_IRUO}/human/images/'
mask_dir = f'{PATH_TO_IRUO}/human/masks/'
plt.rcParams['figure.dpi'] = 250

def xywhtoxyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def ccwhtoxyxy(bbox):
    return list(np.array([bbox[0]-bbox[2]/2, bbox[1]-bbox[3]/2, bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2]).astype(int))

def gate_occlusion2(im: np.ndarray, mask=None, bkgd_mask=None, width=5, spacing=20, mode='horizontal'):
    is_mask = mask is not None
    imsize = im.shape[:2]
    gray = 127 # int(np.amax(im) / 2)
    imtype = im.dtype
    if is_mask:
        masktype = mask.dtype
    try:
        init_row = np.random.randint(0,spacing-width)
    except:
        if is_mask: return None, None
        return None
    out = deepcopy(im)
    outmask = deepcopy(mask)
    outmask_bkgd = deepcopy(bkgd_mask)
    if mode == 'horizontal':
        for a in range(width):
            out[init_row+a:imsize[0]:spacing, :, :] = gray
            if is_mask: 
                outmask[init_row+a:imsize[0]:spacing, :] = 0
                outmask_bkgd[init_row+a:imsize[0]:spacing, :] = 0

    elif mode == 'vertical':
        for a in range(width):
            out[:, init_row+a:imsize[0]:spacing, :] = gray
            outmask[:, init_row+a:imsize[0]:spacing] = 0
            outmask_bkgd[:, init_row+a:imsize[0]:spacing] = 0

    if is_mask: return out.astype(imtype), outmask.astype(imtype), outmask_bkgd.astype(imtype)
    else: return out.astype(imtype)

def gate_occlusion3(im: np.ndarray, mask=None, bkgd_mask=None, widths=(5,0), spacings=(20,0), mode='horizontal'):
    is_mask = mask is not None
    imsize = im.shape[:2]
    gray = 127 # int(np.amax(im) / 2)
    imtype = im.dtype
    if is_mask:
        masktype = mask.dtype
    try:
        init_rows = [np.random.randint(0,s-w) for s, w in zip(spacings, widths)]
    except:
        if (spacings[0] == widths[0] and spacings[1] >= widths[1]) or (spacings[1] == widths[1] and spacings[0] >= widths[0]):
            init_rows = [0, 0]
        else:
            if is_mask: return None, None
            return None
    out = deepcopy(im)
    outmask = deepcopy(mask)
    outmask_bkgd = deepcopy(bkgd_mask)

    occ_mask = np.zeros_like(outmask)
    occ_mask_bkgd = np.zeros_like(outmask_bkgd)

    if mode == 'horizontal':
        for a in range(widths[0]):
            occ_mask[init_rows[0]+a:imsize[0]:spacings[0], :] = 1
        for a in range(widths[1]):
            occ_mask_bkgd[init_rows[1]+a:imsize[1]:spacings[1], :] = 1

    elif mode == 'vertical':
        for a in range(widths[0]):
            occ_mask[:, init_rows[0]+a:imsize[0]:spacings[0]] = 1
        for a in range(widths[1]):
            occ_mask_bkgd[:, init_rows[1]+a:imsize[1]:spacings[1]] = 1

    occ_mask = np.where(outmask>gray, occ_mask, np.zeros_like(occ_mask))
    occ_mask_bkgd = np.where(outmask_bkgd>gray, occ_mask_bkgd, np.zeros_like(occ_mask_bkgd))
    occ_mask_total = np.where(occ_mask+occ_mask_bkgd, np.ones_like(occ_mask), np.zeros_like(occ_mask))
    occ_mask_total_3 = np.repeat(occ_mask_total[:, :, np.newaxis], 3, axis=2)

    out = np.where(occ_mask_total_3, gray*np.ones_like(out), out)
    outmask = np.where(occ_mask_total, np.zeros_like(outmask), outmask)
    outmask_bkgd = np.where(occ_mask_total, np.zeros_like(outmask_bkgd), outmask_bkgd)

    if is_mask: return out.astype(imtype), outmask.astype(imtype), outmask_bkgd.astype(imtype)
    else: return out.astype(imtype)

def xhatch_occlusion2(im: np.ndarray, mask=None, bkgd_mask=None, width=5, spacing=20):
    is_mask = mask is not None       
    out = gate_occlusion2(im, mask=mask, bkgd_mask=bkgd_mask, width=width, spacing=spacing, mode='horizontal')
    if is_mask:
        if out[0] is None: return None, None, None
        out, maskout, maskout_bkgd = out
    else:
        if out is None: return None
    out = gate_occlusion2(out, mask=maskout, bkgd_mask=maskout_bkgd, width=width, spacing=spacing, mode='vertical')
    return out

def xhatch_occlusion3(im: np.ndarray, mask=None, bkgd_mask=None, widths=(5,0), spacings=(20,0)):
    is_mask = mask is not None       
    out = gate_occlusion3(im, mask=mask, bkgd_mask=bkgd_mask, widths=widths, spacings=spacings, mode='horizontal')
    if is_mask:
        if out[0] is None: return None, None, None
        out, maskout, maskout_bkgd = out
    else:
        if out is None: return None
    out = gate_occlusion3(out, mask=maskout, bkgd_mask=maskout_bkgd, widths=widths, spacings=spacings, mode='vertical')
    return out

max_or_zeroish = lambda x: np.amax([0.001, np.amax(x)])
_norm_mask = lambda x: x.astype(float) / max_or_zeroish(x)

def calculate_occlusion_w_mask2(orig_mask, return_mask, bkgd_orig_mask, bkgd_return_mask):
    if orig_mask.ndim > 2:
        orig_mask = orig_mask[...,0]
    if return_mask.ndim > 2:
        return_mask = return_mask[...,0]
    if return_mask is None:
        if orig_mask is None: return -1
        else: return 100
    orig_mask = _norm_mask(orig_mask)
    bkgd_orig_mask = _norm_mask(bkgd_orig_mask)
    return_mask = _norm_mask(return_mask)
    bkgd_return_mask = _norm_mask(bkgd_return_mask)
    # return (object occlusion, background occlusion)
    return (1. - (np.sum(return_mask) / np.sum(orig_mask)), 1. - (np.sum(bkgd_return_mask) / np.sum(bkgd_orig_mask)))

def rotated_gate_occlusion2(im: np.ndarray, mask=None, bkgd_mask=None, angle=45, width=5, spacing=20):
    is_mask = mask is not None
    imsize = im.shape[:2]
    grid_mask = np.zeros(imsize)
    imtype = im.dtype
    if is_mask: masktype = mask.dtype
    try:
        init_row = np.random.randint(0,spacing-width)
    except:
        if is_mask: return None, None
        return None
    for a in range(width):
        grid_mask[init_row+a:imsize[0]:spacing, :] = 1
    grid_mask = cv2.resize(grid_mask, list(((1.42*np.array(imsize)).astype(int))))

    # rotate grid_mask by angle
    grid_mask = ndimage.rotate(grid_mask, angle, reshape=False)
    grid_mask += 0.1 * np.ones_like(grid_mask)
    grid_mask = grid_mask.astype(int)
    #print(grid_mask.shape)

    # then crop image size from the middle
    grmid = (np.array(grid_mask.shape) / 2).astype(int)
    imid = (np.array(imsize) / 2).astype(int)

    grid_mask = grid_mask[grmid[0]-imid[0]:grmid[0]+imid[0], grmid[1]-imid[1]:grmid[1]+imid[1]]

    gray = 127 # int(np.amax(im) / 2)
    out = deepcopy(im)
    outmask = deepcopy(mask)
    outmask_bkgd = deepcopy(bkgd_mask)
    
    grid_mask = np.repeat(grid_mask[:, :, np.newaxis], 3, axis=2).astype(bool)
    out = np.where(grid_mask, gray * np.ones_like(out), out)
    outmask = np.where(grid_mask[...,0], np.zeros_like(outmask), outmask)
    outmask_bkgd = np.where(grid_mask[...,0], np.zeros_like(outmask_bkgd), outmask_bkgd)

    if is_mask: return out.astype(imtype), outmask.astype(masktype), outmask_bkgd.astype(masktype)
    else: return out.astype(imtype)

def rotated_gate_occlusion3(im: np.ndarray, mask=None, bkgd_mask=None, angle=45, widths=(5,0), spacings=(20,0)):
    is_mask = mask is not None
    imsize = im.shape[:2]
    grid_mask = np.zeros(imsize)
    grid_mask_b = np.zeros(imsize)
    imtype = im.dtype
    if is_mask: masktype = mask.dtype
    try:
        init_rows = [np.random.randint(0,s-w) for s, w in zip(spacings, widths)]
    except:
        if (spacings[0] == widths[0] and spacings[1] >= widths[1]) or (spacings[1] == widths[1] and spacings[0] >= widths[0]):
            init_rows = [0, 0]
        else:
            if is_mask: return None, None
            return None
    
    # Object occlusions
    for a in range(widths[0]):
        grid_mask[init_rows[0]+a:imsize[0]:spacings[0], :] = 1

    grid_mask = cv2.resize(grid_mask, list(((1.42*np.array(imsize)).astype(int))))

    # rotate grid_mask by angle
    grid_mask = ndimage.rotate(grid_mask, angle, reshape=False)
    grid_mask += 0.1 * np.ones_like(grid_mask)
    grid_mask = grid_mask.astype(int)
    #print(grid_mask.shape)

    # then crop image size from the middle
    grmid = (np.array(grid_mask.shape) / 2).astype(int)
    imid = (np.array(imsize) / 2).astype(int)

    grid_mask = grid_mask[grmid[0]-imid[0]:grmid[0]+imid[0], grmid[1]-imid[1]:grmid[1]+imid[1]]

    # Background occlusions
    for a in range(widths[1]):
        grid_mask_b[init_rows[1]+a:imsize[1]:spacings[1], :] = 1

    grid_mask_b = cv2.resize(grid_mask_b, list(((1.42*np.array(imsize)).astype(int))))

    # rotate grid_mask by angle
    grid_mask_b = ndimage.rotate(grid_mask_b, angle, reshape=False)
    grid_mask_b += 0.1 * np.ones_like(grid_mask_b)
    grid_mask_b = grid_mask_b.astype(int)
    #print(grid_mask.shape)

    # then crop image size from the middle
    grmid = (np.array(grid_mask_b.shape) / 2).astype(int)
    imid = (np.array(imsize) / 2).astype(int)

    grid_mask_b = grid_mask_b[grmid[0]-imid[0]:grmid[0]+imid[0], grmid[1]-imid[1]:grmid[1]+imid[1]]

    gray = 127 # int(np.amax(im) / 2)
    out = deepcopy(im)
    outmask = deepcopy(mask)
    outmask_bkgd = deepcopy(bkgd_mask)

    grid_mask = np.where(outmask, grid_mask, np.zeros_like(grid_mask))
    grid_mask_b = np.where(outmask_bkgd, grid_mask_b, np.zeros_like(grid_mask))
    
    grid_mask = np.repeat(grid_mask[:, :, np.newaxis], 3, axis=2).astype(bool)
    grid_mask_b = np.repeat(grid_mask_b[:, :, np.newaxis], 3, axis=2).astype(bool)

    out = np.where((grid_mask+grid_mask_b), gray * np.ones_like(out), out)
    outmask = np.where((grid_mask+grid_mask_b)[...,0], np.zeros_like(outmask), outmask)
    outmask_bkgd = np.where((grid_mask+grid_mask_b)[...,0], np.zeros_like(outmask_bkgd), outmask_bkgd)

    if is_mask: return out.astype(imtype), outmask.astype(masktype), outmask_bkgd.astype(masktype)
    else: return out.astype(imtype)

def object_occlude2(image, mask, bkgd_mask, occluder, scale=0.25, angle=0, mode='random', flip=True):
    # occluders have 4 dimensions: take [:,;,3] to figure out where to mask
    # imgs are 224x224; occluders have long dimension of (sqrt(scale) * 224)
    occluder_longdim = int(np.sqrt(scale) * 224)
    comp = int((224 - occluder_longdim) / 2)
    image = cv2.resize(image, dsize=(224,224))
    mask = cv2.resize(mask, dsize=(224,224))
    bkgd_mask = cv2.resize(bkgd_mask, dsize=(224,224))
    occluder = ndimage.rotate(occluder, angle=angle, reshape=False)
    occluder = cv2.resize(occluder, dsize=(occluder_longdim,occluder_longdim))
    if mode == 'center':
        occluder = np.pad(occluder, ((comp,comp),(56,56),(0,0)))
    else:
        randxy = np.random.randint(-(comp-1),comp, size=2)
        occluder = np.pad(occluder, ((comp-randxy[0],comp+randxy[0]),(comp-randxy[1],comp+randxy[1]),(0,0)))
    occluder = cv2.resize(occluder, dsize=(224,224))
    occ_mask = np.stack([occluder[:,:,3] for _ in range(3)], axis=-1)
    occ_mask = occ_mask / np.amax(occ_mask)
    mask = mask / np.amax(mask)
    bkgd_mask = bkgd_mask / np.amax(bkgd_mask)
    occluder = occluder[:,:,:3]
    if flip: occluder[:,:,::-1]
    out = np.where(occ_mask, occluder, image)
    outmask = (255*np.where(occ_mask[...,0], np.zeros_like(mask), mask)).astype(int)
    outmask_bkgd = (255*np.where(occ_mask[...,0], np.zeros_like(bkgd_mask), bkgd_mask)).astype(int)
    return out, outmask, outmask_bkgd

def read_object(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def random_mask(img, mask, bkgd_mask, mode='gray'):
    assert mode in ['black', 'gray', 'white', 'noise', 'texture'], 'Mode must be black/gray/white/noise/texture'
    if mode == 'black': occluder = np.zeros_like(img)
    elif mode == 'gray': occluder = np.ones_like(img) * 127
    elif mode == 'white': occluder = np.ones_like(img) * 255
    elif mode == 'noise': occluder = np.random.randint(0, 255, size=img.shape)
    elif mode == 'texture': occluder = cv2.resize(img, img.shape[:2][::-1])
    #print(bbox_crop.shape)
    art_occ_mask = random_box(img)

    occ_im, outmask, outmask_bkgd = mask_occlude2(img, mask, bkgd_mask, occluder, art_occ_mask)

    return occ_im, outmask, outmask_bkgd

def mask_occlude2(img, mask, bkgd_mask, occluder, occ_mask):
    if np.amax(occ_mask) < 0.001: return None, None, None
    occ_mask = occ_mask / np.amax(occ_mask).astype(int)
    mask = mask / np.amax(mask)
    bkgd_mask = bkgd_mask / np.amax(bkgd_mask)
    occluder = occluder[:,:,:3][:,:,::-1]
    out = np.where(occ_mask, occluder, img)
    outmask = (255*np.where(occ_mask[...,0], np.zeros_like(mask), mask)).astype(int)
    outmask_bkgd = (255*np.where(occ_mask[...,0], np.zeros_like(bkgd_mask), bkgd_mask)).astype(int)
    return out, outmask, outmask_bkgd

def random_box(bbox_crop):
    rb = _rand_box(bbox_crop)
    return _rand_box_mask(bbox_crop, *rb)

def quick_background_mask(mask):
    return np.ones_like(mask) * np.amax(mask) - mask

def diffuse_occlusion4(im: np.ndarray, mask=None, bkgd_mask=None, angle=45, width=5, spacing=20, mode='t'):
    '''
    Mode: 'v' vertical, 'h' horizontal, 't' x-hatch, 'r' rotation (specify angle(s))
    '''
    # Determine mode
    args = (im, width, spacing)
    match mode:
        case 'v':
            o, occluder = _v4(*args)
            mult = spacing / width
        case 'h':
            o, occluder = _h4(*args)
            mult = spacing / width
        case 't':
            o, occluder = _t4(*args)
            denom = width/spacing - width**2/spacing**2
            if denom <= 0: return None, None, None
            mult = 1 / denom
        case 'r':
            try: 
                iter(angle)
                o, occluder = _rr4(im, angle, width, spacing)
                denom = width/spacing - width**2/spacing**2
                if denom <= 0: return None, None, None
                mult = 1 / denom
            except: 
                o, occluder = _r4(im, angle, width, spacing)
                mult = spacing/width
        case _:
            raise ValueError('Must be one of valid types v/h/t/r')
    ''' TRY THIS? '''
    mult = mult if np.random.random() > 0.75 else 1
    # Generate box:
    rbc, rbwh = _rand_box(im)
    #shift = (rbc - np.array(im.shape[:2])) / 2
    #rbc -= shift
    rbwh *= np.sqrt(mult)
    rbc, rbwh = rbc.astype(int), rbwh.astype(int)
    artmask = _rand_box_mask(im, rbc, rbwh)
    artmask = np.where(occluder, artmask, np.zeros_like(artmask))
    return mask_occlude2(im, mask, bkgd_mask, o, artmask)
    
def _h4(im: np.ndarray, width=5, spacing=20):
    imsize = im.shape[:2]
    gray = 127 # int(np.amax(im) / 2)
    out, outmask = np.zeros_like(im), np.zeros_like(im)
    try:
        init_row = np.random.randint(0,spacing-width)
    except:
        return None, None
    #if mode == 'horizontal':
    for a in range(width):
        out[init_row+a:imsize[0]:spacing, :, :] = gray
        outmask[init_row+a:imsize[0]:spacing, :] = 1
    else: return out.astype(int), outmask.astype(int)

def _v4(im: np.ndarray, width=5, spacing=20):
    imsize = im.shape[:2]
    gray = 127 # int(np.amax(im) / 2)
    out, outmask = np.zeros_like(im), np.zeros_like(im)
    try:
        init_row = np.random.randint(0,spacing-width)
    except:
        return None, None
    #elif mode == 'vertical':
    for a in range(width):
        out[:, init_row+a:imsize[0]:spacing, :] = gray
        outmask[:, init_row+a:imsize[0]:spacing, :] = 1
    else: return out.astype(int), outmask.astype(int)

def _t4(im: np.ndarray, width=5, spacing=20):
    h, hm = _h4(im, width, spacing)
    v, vm = _v4(im, width, spacing)
    if h is None or v is None: return None, None
    om = (hm+vm).clip(0,1)
    return (127*om).astype(int), om.astype(int)

def _r4(im: np.ndarray, angle=45, width=5, spacing=20):
    imsize = im.shape
    grid_mask = np.zeros(imsize)
    try:
        init_row = np.random.randint(0,spacing-width)
    except:
        return None, None
    for a in range(width):
        grid_mask[init_row+a:imsize[0]:spacing, :] = 1
    grid_mask = cv2.resize(grid_mask, list(((1.42*np.array(imsize[:2])).astype(int))))

    # rotate grid_mask by angle
    grid_mask = ndimage.rotate(grid_mask, angle, reshape=False)
    grid_mask += 0.1 * np.ones_like(grid_mask)
    grid_mask = grid_mask.astype(int)
    #print(grid_mask.shape)

    # then crop image size from the middle
    grmid = (np.array(grid_mask.shape) / 2).astype(int)
    imid = (np.array(imsize) / 2).astype(int)

    grid_mask = grid_mask[grmid[0]-imid[0]:grmid[0]+imid[0], grmid[1]-imid[1]:grmid[1]+imid[1]]

    gray = 127 # int(np.amax(im) / 2)
    out = gray * np.ones_like(im)
    outmask = np.zeros_like(im)
    
    out = np.where(grid_mask, gray * np.ones_like(out), out)
    outmask = np.where(grid_mask, np.ones_like(outmask), outmask)
    return out.astype(int), outmask.astype(int)

def _rr4(im: np.ndarray, angle=[45,135], width=5, spacing=20):
    h, m1 = _r4(im, angle[0], width, spacing)
    v, m2 = _r4(im, angle[1], width, spacing)
    if h is None or v is None: return None, None
    om = (m1+m2).clip(0,1)
    return (127*om).astype(int), om.astype(int)

def _rand_box(im):
    max_dim = np.amax(im.shape[:2])
    mbox_center = max_dim/4 * np.random.normal(0, 1, 2) + np.array([max_dim, max_dim]) / 2
    randel = np.random.normal(0.5,0.4,2).clip(0,1)
    mbox_wh = (max_dim*np.sqrt(randel)).clip(min=0, max=max_dim)
    return mbox_center, mbox_wh

def _rand_box_mask(im, mbox_center, mbox_wh):
    max_dim = np.amax(im.shape[:2])
    mbox = np.concatenate((mbox_center,mbox_wh))
    mx = np.array(ccwhtoxyxy(mbox)).astype(int)
    mx = mx.clip(min=0, max=max_dim)
    art_occ_mask = np.zeros_like(im)
    art_occ_mask[mx[0]:mx[2], mx[1]:mx[3], :] = np.ones(shape=(mx[2]-mx[0], mx[3]-mx[1],3))
    return art_occ_mask