import pickle, os, sys, time, math
import numpy as np
import pandas as pd
from pymol import cmd
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
    #a,b,c = feat.shape
    feat[feat < -100] = -15
    return feat #.reshape(a, c)

def get_masif_neighbor(pdbid):
    path = prefix+'MasifOutput/01-benchmark_surfaces/'
    mesh = pymesh.load_mesh(path+pdbid+".ply")
    result = np.concatenate([mesh.faces[:, [0,1]], mesh.faces[:, [0,2]], mesh.faces[:, [1,2]]])
    result.sort()
    result = np.unique(result, axis=0)   
    return result

t1 = time.time()

pdbid = sys.argv[1]
center = list(map(float, sys.argv[2].split('_')))
    
outdir = prefix+'AnchorOutput/{}_center_{}_da_{}/'.format(pdbid, sys.argv[2], sys.argv[3])

if os.path.exists(outdir+'am_list.npy'):
    print('masif already exists!')
    exit()
    
t0 = time.time()
anchor_list = np.load(outdir+'anchors.npy', allow_pickle=True)

masif_coords = load_masif_coords(pdbid)
masif_feature = load_masif_feature(pdbid)
nei_list = get_masif_neighbor(pdbid)


masif_feature_coord_nei_dict = {}
am_list = []
for i_rep, anchor_coords in enumerate(anchor_list):
    print('repeat', i_rep)
    
    #  calculate masif feature coord neighbor
    
    dist = pairwise_distances(masif_coords, anchor_coords)   
    idx = (dist.min(axis=1)<6)
    feat = masif_feature[idx]
    coor = masif_coords[idx]
    print("Min dist", dist.min())
    idx_mapping = {}
    c = 0
    for x in np.arange(len(idx)):
        if idx[x]:
            idx_mapping[x] = c
            c += 1
    
    nei_select = []
    for a1, a2 in nei_list:
        if a1 in idx_mapping and a2 in idx_mapping:
            nei_select.append([idx_mapping[a1], idx_mapping[a2]])
    nei_select = np.array(nei_select)

    masif_feature_coord_nei_dict[i_rep] = (feat, coor, nei_select)
    
    # calculate am
    am_dist = pairwise_distances(anchor_coords, coor)
#     sele = np.where(am_dist<=6)
#     i = torch.LongTensor(np.vstack(sele))
#     v = torch.FloatTensor(am_dist[sele])
#     am_sparse = torch.sparse.FloatTensor(i, v, torch.Size([am_dist.shape[0], am_dist.shape[1]]))
    am_list.append(am_dist)
    
    
t1 = time.time()
print('masif feature coord nei am length {}, time {}'.format(len(masif_feature_coord_nei_dict), t1-t0))

with open(outdir+'masif_feature_coord_nei_dict.pk', 'wb') as f:
    pickle.dump(masif_feature_coord_nei_dict, f)
np.save(outdir+'am_list.npy', am_list)

    
    

    
