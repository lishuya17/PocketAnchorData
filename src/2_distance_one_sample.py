import os, pickle, time, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
# from pymol import cmd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import torch


prefix = './'

pdbid = sys.argv[1]
da = int(sys.argv[2])
outdir = prefix+'AnchorOutput/{}_da_{}/'.format(pdbid, da)
    

if os.path.exists(outdir+'at_list.pt'):
    print('distances already exists!')
    exit()
        
anchor_list = np.load(outdir+'anchors.npy', allow_pickle=True)
with open(outdir+'atom_feature.pk', 'rb') as f:
    atom_dict = pickle.load(f)

at_list, aa_list = [], []
for i_rep, anchor_coords in enumerate(anchor_list):
    print('repeat', i_rep)
    
    # at
    t0 = time.time()
    atom_coords = atom_dict[i_rep][1]
    at_dist = pairwise_distances(anchor_coords, atom_coords)
    sele = np.where(at_dist<=6)
    i = torch.LongTensor(np.vstack(sele))
    v = torch.FloatTensor(at_dist[sele])
    at_sparse = torch.sparse.FloatTensor(i, v, torch.Size([at_dist.shape[0], at_dist.shape[1]]))
    at_list.append(at_sparse)
    
    t1 = time.time()
    print('at_distance {} {} s'.format(at_sparse.shape, t1-t0))

    # aa
    aa_dist = pairwise_distances(anchor_coords, anchor_coords)
    sele = np.where(aa_dist<=6)
    i = torch.LongTensor(np.vstack(sele))
    v = torch.FloatTensor(aa_dist[sele])
    aa_sparse = torch.sparse.FloatTensor(i, v, torch.Size([aa_dist.shape[0], aa_dist.shape[1]]))
    aa_list.append(aa_sparse)


    t2 = time.time()
    print('aa_distance {} {} s'.format(aa_sparse.shape, t2-t1))

torch.save(at_list, outdir+'at_list.pt')
torch.save(aa_list, outdir+'aa_list.pt')
