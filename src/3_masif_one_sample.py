import pickle, os, sys, time, math
import numpy as np
import pandas as pd
# from pymol import cmd
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
import torch
import pymesh
from collections import defaultdict

prefix = './'

def load_masif_coords(pdbid):
    path = prefix+'MasifOutput/04a-precomputation_12A/precomputation/'
    x = np.load(path+pdbid+'/p1_X.npy')
    y = np.load(path+pdbid+'/p1_Y.npy')
    z = np.load(path+pdbid+'/p1_Z.npy')
    prt_coord = np.vstack([x,y,z]).transpose()
    return prt_coord

def load_masif_feature(pdbid):
    path = prefix+'MasifOutput/04a-precomputation_12A/precomputation/'
    feat = np.load(path+pdbid+'/p1_input_feat.npy')
    if np.isnan(feat).sum() != 0:
        feat[np.where(np.isnan(feat))] = 0
    feat[feat < -100] = -15
    #a,b,c = feat.shape
    return feat #.reshape(a, c)

def get_masif_neighbor(pdbid):
    path = prefix+'MasifOutput/01-benchmark_surfaces/'
    mesh = pymesh.load_mesh(path+pdbid+".ply")
    result = np.concatenate([mesh.faces[:, [0,1]], mesh.faces[:, [0,2]], mesh.faces[:, [1,2]]])
    result.sort()
    result = np.unique(result, axis=0)
    return result

# neighbor
t1 = time.time()

pdbid = sys.argv[1]
da = sys.argv[2]
outdir = prefix+'AnchorOutput/{}_da_{}/'.format(pdbid, da)
    

if os.path.exists(outdir+'am_list.pt'):
    print('masif already exists!')
    exit()


masif_feature = load_masif_feature(pdbid)
neighbor = get_masif_neighbor(pdbid)
np.save(outdir+'masif_feature.npy', masif_feature)
np.save(outdir+'masif_neighbor.npy', neighbor)


masif_coords = load_masif_coords(pdbid)
anchor_list = np.load(outdir+'anchors.npy',  allow_pickle=True)
am_list = []
for i_rep, anchor_coords in enumerate(anchor_list):
    print('repeat', i_rep)
    
    am_dist = pairwise_distances(anchor_coords, masif_coords)
    sele = np.where(am_dist<=6)
    i = torch.LongTensor(np.vstack(sele))
    v = torch.FloatTensor(am_dist[sele])
    am_sparse = torch.sparse.FloatTensor(i, v, torch.Size([am_dist.shape[0], am_dist.shape[1]]))
    am_list.append(am_sparse)
    
torch.save(am_list, outdir+'am_list.pt')
t2 = time.time()
print('masif feature neighbor and am', t2-t1, 's')