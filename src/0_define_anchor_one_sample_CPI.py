import os, time, pickle, sys, math
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from pymol import cmd
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import pymesh

prefix = './'

def load_pdb(pdbid):    
    cmd.reinitialize()
    cmd.load(prefix+'MasifOutput/01-benchmark_pdbs/{}.pdb'.format(pdbid))
    selName = cmd.get_unused_name("sele_")
    cmd.select(selName, "all")
    exposed = []
    cmd.iterate_state(-1, selName, "exposed.append((x,y,z))", space=locals())
    return np.array(exposed)

def sample_coords(pdbid, start_point, seed=1234, step=1, extend=3, pl_thre=2, pr_thre=4, max_points=800, isprint=False):
    if isprint:
        print('start_point', start_point)
    protein_exposed = load_pdb(pdbid)
    
    coords = np.array([start_point])
    r = 0
    np.random.seed(seed)
    while coords.shape[0] <= max_points:
        if isprint:
            print('round', r, 'start, coords', coords.shape)
        r += 1
        randomness =  (np.random.rand(3)*2-1)*step 
        coord_bond_l = coords.min(axis=0) - extend + randomness
        coord_bond_r = coords.max(axis=0) + extend + randomness
        n_x, n_y, n_z = int((coord_bond_r[0] - coord_bond_l[0])/step), \
                        int((coord_bond_r[1] - coord_bond_l[1])/step), int((coord_bond_r[2] - coord_bond_l[2])/step)
        if isprint:
            print('coord_bond_l', coord_bond_l, 'coord_bond_r', coord_bond_r)
            print('number of initial points', n_x * n_y * n_z, (n_x, n_y, n_z))

        x_coords = np.linspace(coord_bond_l[0], coord_bond_r[0], num = n_x)
        y_coords = np.linspace(coord_bond_l[1], coord_bond_r[1], num = n_y)
        z_coords = np.linspace(coord_bond_l[2], coord_bond_r[2], num = n_z)
        mask = np.ones((n_x, n_y, n_z))
        points = np.where(mask)
        initial_coords = np.array([x_coords[points[0]], y_coords[points[1]], z_coords[points[2]]]).transpose()

        pp_distance = pairwise_distances(protein_exposed, initial_coords)
#         print(pp_distance.min(axis=0))
        pp_distance_filter = (pp_distance.min(axis=0) >= pl_thre) & (pp_distance.min(axis=0) <= pr_thre)
        coords = initial_coords[pp_distance_filter, :]
        if len(coords) == 0:
            return None
        if isprint:
            print('number of in-pocket points', 'coords', coords.shape)
        
        if r>10:
            break
    return coords


if __name__ == '__main__':
    
    pdbid = sys.argv[1]
    center = list(map(float, sys.argv[2].split('_')))
    da = int(sys.argv[3]) # default 4
    repeat = 1
    
    outdir = prefix+'AnchorOutput/{}_center_{}_da_{}/'.format(pdbid, sys.argv[2], sys.argv[3])
    if os.path.exists(outdir+'anchors.npy'.format(pdbid)):
        print('anchor already exists!')
        exit()
    
    t0 = time.time()
    
    np.random.seed(1234)
    seed_list = np.random.randint(0, 10000, repeat*10)
    
    repeat_list = []
    
    for i_rep in range(repeat):
        points = None
        i_try = 0
        while points is None and i_try <= 10:
            points = sample_coords(pdbid, center, seed_list[i_rep*10+i_try], isprint=False)
            i_try += 1
        ac_single = AgglomerativeClustering(n_clusters=None, distance_threshold=2, affinity='euclidean', linkage='single')
        ac_single.fit(points)

        dist_list = []
        for ni in range(ac_single.n_clusters_):
            point_set = points[ac_single.labels_==ni]
            if len(point_set) >= 10:
                point_dist = pairwise_distances([center], point_set).reshape(-1)
                dist_list.append(np.mean(point_dist[np.argsort(point_dist)[:10]]))
            else:
                dist_list.append(99)
        closest_cluster = np.argmin(dist_list)

        new_points = points[ac_single.labels_==closest_cluster]
        ac_complete = AgglomerativeClustering(n_clusters=None, distance_threshold=da, affinity='euclidean', linkage='complete')
        ac_complete.fit(new_points)
        centers = []
        for ni in range(ac_complete.n_clusters_):
            centers.append(np.mean(new_points[ac_complete.labels_==ni], axis=0))
        repeat_list.append(np.array(centers))
    
    t1 = time.time()
    print('anchor generation aug {} shape {} time {} s'.format(repeat, np.array(repeat_list).shape, t1-t0))
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    np.save(outdir+'anchors.npy'.format(pdbid), repeat_list)
