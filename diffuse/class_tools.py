import numpy as np
import pandas as pd
from copy import deepcopy as dc
import os
import sklearn.metrics as skm
import matplotlib.pyplot as plt

# ANIMAL:
####### PERSON 1
####### MAMMAL: 
############## ELEPHANT 2
############## GIRAFFE 3
############## SHEEP 4
############## COW 5
############## MONKEY 6
############## RABBIT 7
############## EQUINE:
##################### HORSE 8
##################### ZEBRA 9
############## BEAR:
##################### GIANT PANDA 10
##################### BEAR (OTHER) 11
############## CARNIVORE:
##################### FELINE:
############################ CAT (OTHER) 12
############################ TIGER 13
##################### CANINE: 
############################ DOG 14
####### BIRD:
############## PARROT 15
############## POULTRY 16
############## BIRD (OTHER) 17
####### FISH 18
####### REPTILE:
############## LIZARD 19
############## TURTLE 20
# VEHICLE:
####### AIRPLANE 21
####### WHEELED VEHICLE:
############## BICYCLE 22
############## MOTORCYCLE 23

CLASSES = [
    'person', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'poultry', 'panda', 'lizard',
    'parrot', 'monkey', 'rabbit', 'tiger', 'fish',
    'turtle', 'bicycle', 'motorcycle', 'airplane'
]

class HierarchicalClass:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent_name = None
        self.depth = None
    def assign_parent(self, parent):
        parent.children.append(self)
        self.parent_name = parent.name
    def __str__(self):
        if len(self.children) > 0:
            return "%s: %s" % (self.name, ', '.join([x.name for x in self.children]))
        else:
            return "%s: (empty)" % self.name

cd = {}

for x in CLASSES:
    cd[x] = HierarchicalClass(x)

for y in ['equine', 'bears', 'feline', 'carnivore', 'birds', 'reptile', 'wheeled_vehicle', 'vehicle', 'animal', 'mammal', 'all']:
    cd[y] = HierarchicalClass(y)

def assign_children(parent, children):
    if type(children) != list:
        children = [children]
    for child in children:
        cd[child].assign_parent(cd[parent])
        
assign_children('mammal', ['elephant', 'giraffe', 'sheep', 'cow', 'monkey', 'rabbit', 'equine', 'bears', 'carnivore'])
assign_children('equine', ['horse', 'zebra'])
assign_children('bears', ['panda', 'bear'])
assign_children('carnivore', ['feline', 'dog'])
assign_children('feline', ['tiger', 'cat'])
assign_children('birds', ['parrot', 'poultry', 'bird'])
assign_children('reptile', ['lizard', 'turtle'])
assign_children('vehicle', ['airplane', 'wheeled_vehicle'])
assign_children('wheeled_vehicle', ['bicycle', 'motorcycle'])
assign_children('animal', ['mammal', 'birds', 'fish', 'reptile'])
assign_children('all', ['animal', 'person', 'vehicle'])

def assign_depth(x, lvl=0):
    cd[x].depth = lvl
    for xy in cd[x].children:
        assign_depth(xy.name, lvl+1)

def list_subs(x, lvl=0):
    print(''.join([' - ' for _ in range(lvl)]), cd[x])
    for xy in cd[x].children:
        list_subs(xy.name, lvl+1)

assign_depth('all')

def acc_row(row):
    answer = row.cls
    lblcols = [c for c in row.index if 'label' in c]
    nr = len(lblcols)
    labels = [row[lc] for lc in lblcols]
    resp = nr - np.sum([pd.isna(labels).astype(int)])
    corr = np.sum(np.array([int(l == answer) for l in labels]).astype(int))
    return corr / resp

def acc_row_super(row):
    answer = row.cls
    lblcols = [c for c in row.index if 'label' in c]
    nr = len(lblcols)
    labels = [row[lc] for lc in lblcols]
    resp = nr - np.sum([pd.isna(labels).astype(int)])
    corr = np.sum(np.array([int(l == answer) for l in labels]).astype(int))
    return 1 if corr / resp > 0.5 else 0

def set_depth(dframe: pd.DataFrame, depth=2, super_mode=False, k=1):
    frame = dc(dframe)
    inlist = dc(CLASSES)
    outlist = dc(CLASSES)

    outdepth = np.array([cd[o].depth for o in outlist])
    while np.sum((outdepth > depth).astype(int)) > 0:
        # replace over-depth items with one step up
        for i, (ol, od) in enumerate(zip(outlist, outdepth)):
            if od > depth:
                outlist[i] = cd[ol].parent_name
        # check new depths
        outdepth = np.array([cd[o].depth for o in outlist])

    for i, o in zip(inlist, outlist):
        frame.cls = frame.cls.str.replace(i, o)
        lbls = [c for c in frame.columns if 'label' in c]
        for lbl in lbls: 
            frame[lbl] = frame[lbl].str.replace(i, o)
    
    frame = frame.drop(columns='acc')
    if super_mode: frame['acc'] = [acc_row_super(x) for x in frame.iloc]
    else: frame['acc'] = [acc_row(x) for x in frame.iloc]
    return frame

def pull_all_responses(nr, exclude=None, img_dir='/Users/Kaleb/iCloudDrive/Documents/Duke/AMLL/amll-occlusion-human-study-win/images/'):
    csvsin = []
    fig, ax = plt.subplots()
    ax.set_xticks([0,1,2])
    ax.set_xlabel('Occlusion level (0-2)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of all respondents (all colors, IDs hidden)')
    for i in range(1,nr+1):
        incsv = pd.read_csv('response%i.csv'%i)
        incsv = incsv.set_index(pd.Index(['r%i-n%i' % (i, nidx) for nidx in range(len(incsv))]))
        incsv.fnames = [x.split('\\')[-1] for x in incsv.fnames.iloc]
        full_images = [x for x in os.listdir(img_dir) if '.jpg' in x]
        incsv = incsv[incsv.fnames.isin(full_images)]
        if exclude is not None:
            if i not in exclude: 
                csvsin.append(incsv)
        else: csvsin.append(incsv)
        incsv_0 = incsv[incsv.occ_level==0]
        incsv_1 = incsv[incsv.occ_level==1]
        incsv_2 = incsv[incsv.occ_level==2]
        print('Response %i:' % i)
        acc = lambda x: np.sum(x.survey_labels == x.cls)/len(x)
        print('0:', np.sum(incsv_0.survey_labels == incsv_0.cls), 'of', len(incsv_0), acc(incsv_0))
        print('1:', np.sum(incsv_1.survey_labels == incsv_1.cls), 'of', len(incsv_1), acc(incsv_1))
        print('2:', np.sum(incsv_2.survey_labels == incsv_2.cls), 'of', len(incsv_2), acc(incsv_2))
        ax.plot([acc(i) for i in [incsv_0, incsv_1, incsv_2]], label='response%i'%i, marker='.')
        print('------------------------')
    csvsin = pd.concat(csvsin)
    try:
        csvsin.drop(columns='num_times_seen', inplace=True)
    except: pass
    return csvsin

def print_total_accs(csvsin):
    occ_0 = csvsin[csvsin.occ_level == 0]
    occ_1 = csvsin[csvsin.occ_level == 1]
    occ_2 = csvsin[csvsin.occ_level == 2]
    for i, x in enumerate([occ_0, occ_1, occ_2]):
        print('occ %i' % i, np.sum(x.cls == x.survey_labels) / len(x))

def generate_answer_table(csvsin, nr):
    answer_table = pd.DataFrame(columns=['fnames', 'cls', 'occ_level', *['label%i' % (x+1) for x in range(nr)]])
    for n, idx in enumerate(pd.value_counts(csvsin.fnames).index):
        # get each participant label
        filter = csvsin[csvsin.fnames == idx]
        fcls = filter.cls.iloc[0]
        focc = filter.occ_level.iloc[0]
        labels = []
        for fidx, f in enumerate(filter.survey_labels.iloc):
            labels.append(f)
        for _ in range(nr - len(filter)):
            labels.append(None)
        newline_dict = {}
        newline_dict['fnames'] = idx
        newline_dict['cls'] = fcls
        newline_dict['occ_level'] = focc
        for fidx in range(nr):
            newline_dict['label%i' % (fidx+1)] = labels[fidx]
        newline_dict = pd.DataFrame(newline_dict, columns=newline_dict.keys(), index=[n])
        if len(newline_dict) > 0:
            answer_table = pd.concat([answer_table, newline_dict])
    answer_table['acc'] = [acc_row(x) for x in answer_table.iloc]
    return answer_table

def print_accs(answer_table, verbose=True, override=None):
    if override is not None:
        occ_0_acc = np.average(answer_table[answer_table.occ_level == 0][override])
        occ_1_acc = np.average(answer_table[answer_table.occ_level == 1][override])
        occ_2_acc = np.average(answer_table[answer_table.occ_level == 2][override])
    else:
        occ_0_acc = np.average(answer_table[answer_table.occ_level == 0].acc)
        occ_1_acc = np.average(answer_table[answer_table.occ_level == 1].acc)
        occ_2_acc = np.average(answer_table[answer_table.occ_level == 2].acc)
    if verbose: print('0:', occ_0_acc, '1:', occ_1_acc, '2:', occ_2_acc)
    return [occ_0_acc, occ_1_acc, occ_2_acc]

'''def print_skm_accs(answer_table):
    occ_0_acc = skm.accuracy_score(answer_table[answer_table.occ_level == 0])
    occ_1_acc = np.average(answer_table[answer_table.occ_level == 1].acc)
    occ_2_acc = np.average(answer_table[answer_table.occ_level == 2].acc)
    print('0:', occ_0_acc, '1:', occ_1_acc, '2:', occ_2_acc)
    return [occ_0_acc, occ_1_acc, occ_2_acc]'''

'''full_images = [x for x in os.listdir('/Users/Kaleb/iCloudDrive/Documents/Duke/AMLL/amll-occlusion-human-study-win/images/') if '.jpg' in x]
for x in full_images:
    if str(x) not in answer_table.fnames.iloc:
        newline_dict = {}
        newline_dict['fnames'] = x
        newline_dict['cls'] = x.split('_')[0]
        newline_dict['occ_level'] = x.split('_')[1].replace('o','')
        for fidx in range(nr):
            newline_dict['label%i' % (fidx+1)] = None
        answer_table = answer_table.append(newline_dict, ignore_index=True)'''