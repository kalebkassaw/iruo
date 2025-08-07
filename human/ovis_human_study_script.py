# %%
import numpy as np
import pandas as pd
import cv2
import os
import json
from tqdm import tqdm
from ovis_tools import CLASSES
import shutil

PATH_TO_IRUO = ""
# %%
original_csv = pd.read_csv('hstudimages_blur_removed/log.csv')
original_csv['cls'] = [x.split('/')[6] for x in original_csv.src.iloc]
original_csv['occ_level'] = [int(x.split('/')[4]) for x in original_csv.src.iloc]
print(original_csv)
print(original_csv.src.value_counts)

# %%
# 'f'{PATH_TO_IRUO}/images/0/val'
fstruct = 'f'{PATH_TO_IRUO}/images/%i/val/%s' # % (ol, cls)
np.random.seed(919)
get_image = lambda x: cv2.resize(cv2.imread(x), [224,224])[:,:,::-1]
get_image_realsize = lambda x: cv2.imread(x)[:,:,::-1]

# %%
cats = []
lens = [[],[],[]]

for category in CLASSES[:23]:
    cats.append(category)
    for occlusion_level in range(3):
        lens[occlusion_level].append(int(len(os.listdir(fstruct % (occlusion_level, category)))))

df = pd.DataFrame(np.array([cats, *lens]).T, columns=['category', 'images_0', 'images_1', 'images_2'])
display(df)

df.images_0 = df.images_0.astype(int)
df.images_1 = df.images_1.astype(int)
df.images_2 = df.images_2.astype(int)
#count_humans = np.clip(np.round(np.array(df)[:,1:].astype(int) / 50, 0), np.ones_like(np.array(df)[:,1:]), None)
count_humans = (8 * np.ones_like(np.array(df)[:,1:])).astype(int)

df2 = pd.DataFrame(np.concatenate([np.expand_dims(df.category, axis=-1),count_humans.astype(int)], axis=1), columns=df.columns)
display(df2)

# %%
srcs = []
dsts = []
lvars = []
dims = []
count_humans = count_humans.astype(int)
print(count_humans.shape)
for clidx, i in enumerate(count_humans):
    for ol, j in enumerate(count_humans[clidx]):
        # i cls ;; j occl level
        old_match = original_csv[original_csv.occ_level == ol]
        old_match = old_match[old_match.cls == CLASSES[clidx]]
        old_match = old_match[old_match.px > 100**2]
        old_match = old_match[old_match.lvar > 20]
        #print(old_match)
        reps = j
        imgs = list(os.listdir(fstruct % (ol, CLASSES[clidx])))
        print(j, clidx, ol)
        n=0
        dup_flag = False
        while n<reps:
            # if there is an image available, choose it
            if n<(len(old_match)):
                if dup_flag: 
                    src = os.path.join(fstruct % (ol, CLASSES[clidx]), np.random.choice(imgs))
                    dup_flag = False
                else: src = old_match.src.iloc[n]
            # else, pick random image
            else:
                src = os.path.join(fstruct % (ol, CLASSES[clidx]), np.random.choice(imgs))
            if src not in srcs:
                srcs.append(src)
                dst = 'hstudimages_blur_removed_bal/%s_o%i_n%i.jpg' % (CLASSES[clidx], ol, n)
                dsts.append(dst)
                current_im = get_image(src)
                lvar = cv2.Laplacian(current_im, cv2.CV_64F).var()
                dim = np.prod(get_image_realsize(src).shape[:2])
                dims.append(dim)
                lvars.append(lvar)
                if (lvar > 20) and (dim > (100**2)): 
                    shutil.copy2(src, dst)
                    n += 1
                else: 
                    shutil.copy2(src, dst.replace('_removed', 'ry'))
            else:
                dup_flag = True
            
# %%
pd.DataFrame(np.array([srcs, dsts, lvars, dims]).T, columns=['src', 'dst', 'lvar', 'px']).to_csv('hstudimages_blur_removed_bal/log.csv')
# %%
df_summary = pd.read_csv('hstudimages_blur_removed_bal/log.csv')
# %%
print(len(df_summary))
# %%
print(pd.value_counts(df_summary.src))
# %%
print(len(os.listdir('hstudimages_blur_removed')))
# %%
print(df_summary)
# %%
print(current_im.shape)
# %%
