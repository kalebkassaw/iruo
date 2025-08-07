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

OVIS_PATH = ""
PATH_TO_IRUO = ""

sample_rate = 1 # 10
print('Sampling every 1/%i images.' % sample_rate)

#os.chdir(THIS_DIR)

OVIS_PATH = "" # CHANGE

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
                #if o > 0: continue # skip already-occluded images?
                #color = i * 255 / len(bboxs)
                im = read_image(f'{OVIS_PATH}/train/%s' % f, False)
                cropbox = np.array([b[0], b[1], b[0]+b[2], b[1]+b[3]]).astype(int)
                #print(sm, sm.shape, np.amin(sm), np.amax(sm))
                #bbox_crop = crop_image(im, cropbox)
                sm_crop = crop_2d_mask(sm, cropbox)
                partition = 'val' if vid_id > split_number(o, CLASSES[cat_id]) else 'train'
                #master_save_loc = '%s/%s/%s/%s/%i_%05d_%s_%s' % ('%s','%i',partition,CLASSES[cat_id],vid_id,i,'%i',f.replace('/','_'))

                ######
                # Segmentation masks
                ######

                try:
                    save_loc = '%s/%i/%s/%s/%i_%05d_%s' % ('seg-masks', o, partition ,CLASSES[cat_id],vid_id,i,f.replace('/','_'))
                    loud_imwrite(os.path.join(save_dir, save_loc), 255*sm_crop)
                    #print(sm_crop, sm_crop.shape, cropbox, np.amin(sm_crop), np.amax(sm_crop))
                    #print(save_loc)
                except: pass
                #if i > 2: 1/0