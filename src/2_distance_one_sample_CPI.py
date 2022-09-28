import os, pickle, time, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from pymol import cmd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import torch


pdbid = sys.argv[1]
center = list(map(float, sys.argv[2].split('_')))
    
prefix = './'
outdir = prefix+'AnchorOutput/{}_center_{}_da_{}/'.format(pdbid, sys.argv[2], sys.argv[3])
if os.path.exists(outdir+'at_list.npy'):
    print('distances already exists!')
    exit()
    
t0 = time.time()

anchor_list = np.load(outdir+'anchors.npy', allow_pickle=True)
with open(outdir+'atom_feature.pk', 'rb') as f:
    atom_dict = pickle.load(f)

at_list, aa_list = [], []
for i_rep, anchor_coords in enumerate(anchor_list):
    print('repeat', i_rep)
    # at
    atom_coords = atom_dict[i_rep][1]
    at_dist = pairwise_distances(anchor_coords, atom_coords)
#     sele = np.where(at_dist<=6)
#     i = torch.LongTensor(np.vstack(sele))
#     v = torch.FloatTensor(at_dist[sele])
#     at_sparse = torch.sparse.FloatTensor(i, v, torch.Size([at_dist.shape[0], at_dist.shape[1]]))
    at_list.append(at_dist)
    t1 = time.time()
    print('at_distance {} {} s'.format(at_dist.shape, t1-t0))

    # aa
    aa_dist = pairwise_distances(anchor_coords, anchor_coords)
#     sele = np.where(aa_dist<=6)
#     i = torch.LongTensor(np.vstack(sele))
#     v = torch.FloatTensor(aa_dist[sele])
#     aa_sparse = torch.sparse.FloatTensor(i, v, torch.Size([aa_dist.shape[0], aa_dist.shape[1]]))
    aa_list.append(aa_dist)
    t2 = time.time()
    print('aa_distance {} {} s'.format(aa_dist.shape, t2-t1))

np.save(outdir+'at_list.npy', at_list)
np.save(outdir+'aa_list.npy', aa_list)
