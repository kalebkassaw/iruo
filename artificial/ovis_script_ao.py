# %%
import numpy as np
import pandas as pd
import cv2
import os
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

import cocoapi.PythonAPI.pycocotools.ytvos as y
import cocoapi.PythonAPI.pycocotools as pct
import cocoapi.PythonAPI.pycocotools.mask as pctm
from ovis_tools import *

np.random.seed(919)
sample_rate = 1 # 10
print('Sampling every 1/%i images.' % sample_rate)

#os.chdir(THIS_DIR)
OVIS_PATH = ""
PATH_TO_IRUO = ""

train_anno = f'{OVIS_PATH}/annotations_train.json'
#test_anno = f'{OVIS_PATH}/annotations_valid.json'

train_img = f'{OVIS_PATH}/annotations_train.json'

train_val_split = pd.read_csv('train_val_split.csv', delimiter=' ', names=['occ_level', 'cls', 'split_num', 'train_part', 'val_part'])

def split_number(occ_level, cls):
    ud = train_val_split[train_val_split.occ_level == occ_level]
    ud = ud[ud.cls == cls]
    return int(ud.split_num.iloc[0])

def loud_imwrite(path, im):
    if not cv2.imwrite(path, im):
        print("imwrite failed at %s :(" % (path))

# %%
with open(train_anno) as rf: dset = dict(json.load(rf))
yt = y.YTVOS(train_anno)
keys = yt.loadAnns(1)[0].keys()

# %%
# Iterate over objects.
#print('RUNNING FROM 755 ON... IF DOING THIS OVER, THAT NEEDS TO BE FIXED IN LINE 45')
pbar = tqdm(range(len(yt.anns)))

save_dir = f'{PATH_TO_IRUO}/'
for a in pbar:
    anns = yt.loadAnns(a+1)[0]
    # Iterate over annotations (for objects).
    vid_id = anns['video_id']-1 
    cat_id = anns['category_id']-1
    vid_id = anns['video_id']-1
    xid = anns['id']
    bboxs = anns['bboxes']
    occls = anns['occlusion']
    fms = dset['videos'][vid_id]['file_names']
    if None in anns['segmentations']:
        missing_samples = []
        for i, seg_sample in enumerate(anns['segmentations']):
            if seg_sample is None:
                missing_samples.append(i)
        annssegm = [amn is not None for amn in anns['segmentations']]
        anns['segmentations'] = np.array(anns['segmentations'])
        anns['segmentations'] = anns['segmentations'][annssegm]
        anns['segmentations'] = anns['segmentations'].tolist()
        seg_masks = pctm.decode(anns['segmentations'])
        for j in missing_samples:
            #anns['segmentations'].insert(j, None)
            anns['segmentations'] = np.insert(anns['segmentations'], j, None)
    else:
        seg_masks = pctm.decode(anns['segmentations'])
    seg_masks = np.transpose(seg_masks, (2,0,1))
    for i, (b, f, o, sm) in enumerate(zip(bboxs, fms, occls, seg_masks)):
        if i % sample_rate != 0 or sm is None:
            continue
        if b is not None:
            if cat_id <= 22: # knock out boat, vehicle
                o = occlusion_num(o)
                if o > 0: continue # skip already-occluded images?
                #color = i * 255 / len(bboxs)
                im = read_image(f'{OVIS_PATH}/train/%s' % f, False)
                '''cropbox = np.array([b[0], b[1], b[0]+b[2], b[1]+b[3]]).astype(int)
                bbox_crop = crop_image(im, cropbox)'''
                partition = 'val' if vid_id > split_number(o, CLASSES[cat_id]) else 'train'
                master_save_loc = '%s/%s/%s/%s/%i_%05d_%s_%s' % ('%s','%i',partition,CLASSES[cat_id],vid_id,i,'%i',f.replace('/','_'))

                # TODO: more!!!!!
                occ_iter_idx = 1
                #for occ_iter_idx in range(3):

                ######
                # Black masks
                ######
                try:
                    occ_im, pct_occ = random_mask(im, b, sm, 'black')
                    save_loc = master_save_loc % ('images-black', occ_to_ovis_level(pct_occ), occ_iter_idx)
                    loud_imwrite(os.path.join(save_dir, save_loc), occ_im)
                except: pass

                ######
                # White masks
                ######
                try:
                    occ_im, pct_occ = random_mask(im, b, sm, 'white')
                    save_loc = master_save_loc % ('images-white', occ_to_ovis_level(pct_occ), occ_iter_idx)
                    loud_imwrite(os.path.join(save_dir, save_loc), occ_im)
                except: pass

                ######
                # Noise masks
                ######
                try:
                    occ_im, pct_occ = random_mask(im, b, sm, 'noise')
                    save_loc = master_save_loc % ('images-noise', occ_to_ovis_level(pct_occ), occ_iter_idx)
                    loud_imwrite(os.path.join(save_dir, save_loc), occ_im)
                except: pass

                ######
                # Zebra print masks
                ######
                try:
                    occ_im, pct_occ = random_mask(im, b, sm, 'texture')
                    save_loc = master_save_loc % ('images-texture', occ_to_ovis_level(pct_occ), occ_iter_idx)
                    loud_imwrite(os.path.join(save_dir, save_loc), occ_im)
                except: pass
                ######
                #Object masks!
                ######
                try:
                    rand_objects = [os.path.join('object_images', x) for x in os.listdir('object_images')]
                    occl_im_in = read_image(np.random.choice(rand_objects), False)

                    bbox_crop = crop_image_dimat2(im, xywhtoxyxy(b))
                    
                    objmask = np.stack([sm,sm,sm],axis=-1)
                    objmask = crop_image_dimat2(objmask, xywhtoxyxy(b))
                    
                    occ_im, pct_occ = object_occlude(bbox_crop, occl_im_in, objmask, 'random')
                    save_loc = master_save_loc % ('images-object',occ_to_ovis_level(pct_occ), occ_iter_idx)
                    loud_imwrite(os.path.join(save_dir, save_loc), occ_im)
                except: pass