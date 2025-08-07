import numpy as np
import cv2
import torch
import os

#os.chdir(THIS_DIR)

def occlusion_num(occlusion_level):
    levels = {'no_occlusion': 0,
    'slight_occlusion': 1,
    'severe_occlusion': 2}
    return levels[occlusion_level]

def occ_to_ovis_level(prop):
    if prop < 0.0001:
        return 0
    elif prop < 0.5:
        return 1
    else:
        return 2

def xywhtoxyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def crop_image(image, bbox, expansion=20, return_bbox=False):
    #bbox = xywhtoxyxy(bbox)
    imsize = np.shape(image)
    xsize = bbox[2] - bbox[0]
    ysize = bbox[3] - bbox[1]
    if xsize > ysize:
        bbox[1] -= int((xsize - ysize) / 2)
        bbox[3] += int((xsize - ysize) / 2)
    elif ysize > xsize:
        bbox[0] -= int((ysize - xsize) / 2)
        bbox[2] += int((ysize - xsize) / 2)
    xmin = np.amax([bbox[0] - expansion, 0])
    xmax = np.amin([bbox[2] + expansion, imsize[1]])
    ymin = np.amax([bbox[1] - expansion, 0])
    ymax = np.amin([bbox[3] + expansion, imsize[2]])

    # TODO: probably need a way to handle non-square aspect ratios if at all possible
    
    # print(xmin, xmax, ymin, ymax)
    crop = image[:, xmin:xmax, ymin:ymax]

    if return_bbox: return crop, bbox
    else: return crop

def crop_2d_mask(image, bbox, expansion=20, return_bbox=False):
    #bbox = xywhtoxyxy(bbox)
    imsize = np.shape(image)
    xsize = bbox[2] - bbox[0]
    ysize = bbox[3] - bbox[1]
    if xsize > ysize:
        bbox[1] -= int((xsize - ysize) / 2)
        bbox[3] += int((xsize - ysize) / 2)
    elif ysize > xsize:
        bbox[0] -= int((ysize - xsize) / 2)
        bbox[2] += int((ysize - xsize) / 2)
    xmin = np.amax([bbox[0] - expansion, 0])
    xmax = np.amin([bbox[2] + expansion, imsize[1]])
    ymin = np.amax([bbox[1] - expansion, 0])
    ymax = np.amin([bbox[3] + expansion, imsize[0]])

    # TODO: probably need a way to handle non-square aspect ratios if at all possible
    
    # print(xmin, xmax, ymin, ymax)
    crop = image[ymin:ymax, xmin:xmax]

    if return_bbox: return crop, bbox
    else: return crop

def crop_image_dimat2(image, bbox, expansion=20, return_bbox=False):
    #bbox = xywhtoxyxy(bbox)
    imsize = np.shape(image)
    xsize = bbox[2] - bbox[0]
    ysize = bbox[3] - bbox[1]
    if xsize > ysize:
        bbox[1] -= int((xsize - ysize) / 2)
        bbox[3] += int((xsize - ysize) / 2)
    elif ysize > xsize:
        bbox[0] -= int((ysize - xsize) / 2)
        bbox[2] += int((ysize - xsize) / 2)
    xmin = np.amax([bbox[0] - expansion, 0])
    xmax = np.amin([bbox[2] + expansion, imsize[1]])
    ymin = np.amax([bbox[1] - expansion, 0])
    ymax = np.amin([bbox[3] + expansion, imsize[0]])

    # TODO: probably need a way to handle non-square aspect ratios if at all possible
    
    #print(xmin, xmax, ymin, ymax)
    crop = image[ymin:ymax, xmin:xmax, :]

    if return_bbox: return crop, bbox
    else: return crop

def mask_occlude(bbox_crop, objmask, occluder, art_occ_mask):
    occluded_image = np.where(art_occ_mask, occluder, bbox_crop)
    overlap = np.sum(np.where(art_occ_mask+objmask==2, 1, 0))
    object_area = np.sum(objmask)
    pct_occ = overlap / object_area
    return occluded_image, pct_occ

def object_occlude(image, occluder, objmask, mode='random'):
    # occluders have 4 dimensions: take [:,;,3] to figure out where to mask
    # imgs are 224x224; occluders have long dimension of 112
    image = cv2.resize(image, dsize=(224,224))
    objmask = cv2.resize(objmask, dsize=(224,224))
    occluder = cv2.resize(occluder, dsize=(112,112))
    if mode == 'center':
        occluder = np.pad(occluder, ((56,56),(56,56),(0,0)))
    else:
        randxy = np.random.randint(-55,55+1, size=2)
        #TODO: swap [0] and [1] back if this is broken?
        occluder = np.pad(occluder, ((56-randxy[0],56+randxy[0]),(56-randxy[1],56+randxy[1]),(0,0)))
    mask3d = np.stack([occluder[:,:,3] for _ in range(3)], axis=-1)
    mask3d = mask3d / np.amax(mask3d)
    objmask = objmask / np.amax(objmask)
    occluder = occluder[:,:,:3]#[:,:,::-1]
    out = np.where(mask3d, occluder, image)
    overlap = np.sum(np.where(mask3d+objmask==2, 1, 0))
    object_area = np.sum(objmask)
    pct_occ = overlap / object_area
    return out, pct_occ

def random_box(bbox_crop):
    mbox_xy = np.random.randint(30,np.amax(bbox_crop.shape[:2])-30, 2)
    mbox_xy = np.clip(mbox_xy, a_min=np.zeros_like(mbox_xy), a_max=bbox_crop.shape[:2])
    mbox_wh = np.random.randint(50,np.amax(bbox_crop.shape[:2]), 2)
    mbox_wh = np.clip(mbox_wh, a_min=np.zeros_like(mbox_wh), a_max=bbox_crop.shape[:2]-mbox_xy)
    mbox = np.concatenate((mbox_xy,mbox_wh))
    mx = np.array(xywhtoxyxy(mbox)).astype(int)
    #print('mask box', mx)
    art_occ_mask = np.zeros_like(bbox_crop)
    #print('photo', art_occ_mask.shape)
    art_occ_mask[mx[0]:mx[2], mx[1]:mx[3], :] = np.ones(shape=(mbox[2],mbox[3],3))
    return art_occ_mask

def random_mask(img, object_bbox, object_mask, mode='black'):
    b = object_bbox
    m = object_mask
    
    bbox_crop = crop_image_dimat2(img, xywhtoxyxy(b))
    objmask = np.stack([m,m,m],axis=-1)
    objmask = crop_image_dimat2(objmask, xywhtoxyxy(b))

    assert mode in ['black', 'white', 'noise', 'texture'], 'Mode must be black, white, noise, or texture'
    if mode == 'black': occluder = np.zeros_like(bbox_crop)
    elif mode == 'white': occluder = np.ones_like(bbox_crop) * 255
    elif mode == 'noise': occluder = np.random.randint(0, 255, size=bbox_crop.shape)
    elif mode == 'texture': occluder = cv2.resize(zebra_texture, bbox_crop.shape[:2][::-1])
    #print(bbox_crop.shape)
    art_occ_mask = random_box(bbox_crop)

    occ_im, pct_occ = mask_occlude(bbox_crop, objmask, occluder, art_occ_mask)

    return occ_im, pct_occ

def read_image(path, torchcvt=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = np.stack([img,img,img], axis=-1)
    if img.shape[2] == 3:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    if torchcvt: 
        img = np.transpose(img, (2,0,1))
        img = torch.Tensor(img)
    return img

zebra_texture = read_image('istockphoto-1181378242-612x612.jpg', False)

CLASSES = [
    'person', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'poultry', 'giant_panda', 'lizard',
    'parrot', 'monkey', 'rabbit', 'tiger', 'fish',
    'turtle', 'bicycle', 'motorcycle', 'airplane',
    'boat', #REMOVE
    'vehicle' #REMOVE
]