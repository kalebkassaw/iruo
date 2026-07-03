# %%
import numpy as np
import pandas as pd
import cv2
import os
import json
from tqdm import tqdm

OVIS_PATH = ""
PATH_TO_IRUO = ""

# %%
def xywhtoxyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def crop_image(image, bbox, expansion=20):
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
    return crop

train_anno = f'{OVIS_PATH}/annotations_train.json'
#test_anno = f'{OVIS_PATH}/annotations_valid.json'

train_img = f'{OVIS_PATH}/annotations_train.json'

def occlusion_num(occlusion_level):
    levels = {'no_occlusion': 0,
    'slight_occlusion': 1,
    'severe_occlusion': 2}
    return levels[occlusion_level]

# %%
with open(train_anno) as rf: dset = dict(json.load(rf))

# %%
print(dset.keys())

# %%
print([i['name'] for i in dset['categories']])

# %%
#print(train['annotations'][0])

# %%
print(len(dset['videos']))

# %%
#print(train['annotations'][0])

# %%
import cocoapi.PythonAPI.pycocotools.ytvos as y

# %%
yt = y.YTVOS(train_anno)

# %%
#print(len(yt.anns))

# %%
keys = yt.loadAnns(1)[0].keys()

# %%
#print(keys)

# %%
#aanns = yt.loadAnns(1 or 2 or 910)[0]
aanns = yt.loadAnns(3)[0]
cat_id = aanns['category_id']
vid_id = aanns['video_id']
id = aanns['id']
bboxs = aanns['bboxes']
occls = aanns['occlusion']
#print(occls)

# %%
'''print(cat_id)
print(vid_id)
print(id)'''
'''print(bboxs)
print(occls)'''

CLASSES = [
    'person', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'poultry', 'giant_panda', 'lizard',
    'parrot', 'monkey', 'rabbit', 'tiger', 'fish',
    'turtle', 'bicycle', 'motorcycle', 'airplane',
    'boat', #REMOVE
    'vehicle' #REMOVE
]
#len(CLASSES)

# %%
print(CLASSES[23])

# %%
#1::85aa3b0e #2::fe6a535a
#val_threshold = 480
save_dir = f'{PATH_TO_IRUO}/images/'
pbar = tqdm(range(len(yt.anns)), desc='Making OVIS train crops')
for a in pbar:
    aanns = yt.loadAnns(a+1)[0] # adjustment for mat indexing
    cat_id = aanns['category_id']-1 # adjustment for mat indexing
    vid_id = aanns['video_id']-1 # adjustment for mat indexing
    id = aanns['id']
    bboxs = aanns['bboxes']
    occls = aanns['occlusion']
    fms = dset['videos'][vid_id]['file_names']
    #partition = 'val' if vid_id > val_threshold else 'train'
    pbar.set_description('Making OVIS %s crops' % 'partition')
    for i, (b, f, o) in enumerate(zip(bboxs, fms, occls)):
        if b is not None:
            if cat_id > 22: # knock out boat, vehicle
                o = occlusion_num(o)
                #color = i * 255 / len(bboxs)
                im = cv2.imread(f'{OVIS_PATH}/train/%s' % f).T
                cropbox = np.array([b[0], b[1], b[0]+b[2], b[1]+b[3]]).astype(int)
                crop = crop_image(im, cropbox)
                save_loc = '%i/%s/%s/%i_%05d_%s' % (o,'partition',CLASSES[cat_id],vid_id,i,f.replace('/','_'))
                if not cv2.imwrite(save_dir + save_loc, crop.T):
                    print('fail', save_loc)
