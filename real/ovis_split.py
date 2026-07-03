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

train_anno = 'f'{OVIS_PATH}/annotations_train.json'
#test_anno = 'f'{OVIS_PATH}/annotations_valid.json'

train_img = 'f'{OVIS_PATH}/annotations_train.json'

def occlusion_num(occlusion_level):
    levels = {'no_occlusion': 0,
    'slight_occlusion': 1,
    'severe_occlusion': 2}
    return levels[occlusion_level]

CLASSES = [
    'person', 
    'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'poultry', 'giant_panda', 'lizard',
    'parrot', 'monkey', 'rabbit', 'tiger', 'fish',
    'turtle', 'bicycle', 'motorcycle', 'airplane'
]

# %%
image_dist = np.zeros((3, len(CLASSES)))
split_ratio_train = np.zeros((3, len(CLASSES)))
split_ratio_val = np.zeros((3, len(CLASSES)))
save_dir = f'{PATH_TO_IRUO}/images/'
with open(train_anno) as rf: dset = dict(json.load(rf))
NUM_VIDS = len(dset['videos'])
for ol in range(3):
    for j, clsname in enumerate(CLASSES):
        vid_index = np.zeros(NUM_VIDS)
        for fname in os.listdir(os.path.join(save_dir, '%i/partition/%s' % (ol, clsname))):
            vid_num = int(fname.split('/')[-1].split('_')[0])
            vid_index[vid_num] += 1
        vid_index_s = np.cumsum(vid_index)
        vid_index_s /= vid_index_s[-1]
        vid_index_s -= 0.7 * np.ones_like(vid_index_s)
        vid_index_s = np.abs(vid_index_s)
        image_dist[ol, j] = np.argmin(vid_index_s)
        split_ratio_train[ol, j] = np.cumsum(vid_index)[int(image_dist[ol, j])]
        split_ratio_val[ol, j] = np.cumsum(vid_index)[-1] - int(split_ratio_train[ol, j])

image_dist = image_dist.astype(int)

for afafafaf in range(3):
    for ccc, cn in enumerate(CLASSES):
        print(afafafaf, cn, image_dist[afafafaf,ccc], split_ratio_train[afafafaf,ccc], split_ratio_val[afafafaf,ccc])

# %%
means = []
anno_file_train = [[], [], []]
anno_file_val = [[], [], []]
for ol in range(3):
    for j, clsname in enumerate(tqdm(CLASSES, desc='Splitting at occlusion level %i' % ol)):
        for fname in os.listdir(os.path.join(save_dir, '%i/partition/%s' % (ol, clsname))):
            im = cv2.imread(os.path.join(save_dir, '%i/partition/%s/%s' % (ol, clsname, str(fname))))
            if ol == 0: means.append(np.mean(im, axis=(0,1)))
            vid_num = int(fname.split('/')[-1].split('_')[0])
            if vid_num > image_dist[ol,j]:
                if cv2.imwrite(os.path.join(save_dir, '%i/val/%s/%s' % (ol, clsname, str(fname))), im):
                    anno_file_val[ol].append('%i/val/%s/%s %i\n' % (ol, clsname, str(fname), j))
                else:
                    print('fail', '%i/val/%s/%s %i\n' % (ol, clsname, str(fname), j))
            else:
                if cv2.imwrite(os.path.join(save_dir, '%i/train/%s/%s' % (ol, clsname, str(fname))), im):
                    anno_file_train[ol].append('%i/train/%s/%s %i\n' % (ol, clsname, str(fname), j))
                else:
                    print('fail', '%i/train/%s/%s %i\n' % (ol, clsname, str(fname), j))


# %%
means = np.array(means)
print(np.mean(means, axis=0))
print(np.std(means, axis=0))

# %%
for ou in range(3):
    with open(f'{PATH_TO_IRUO}/annos/train%i.txt' % ou, 'w') as wf:
        wf.writelines(anno_file_train[ou])
    with open(f'{PATH_TO_IRUO}/annos/val%i.txt' % ou, 'w') as wf:
        wf.writelines(anno_file_val[ou])