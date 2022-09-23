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
tmp_dir = './tmp/'
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

def load_protein_coords(pdbid, chains, removeHs=True):
    if os.path.exists(prefix+'MasifOutput/00-raw_pdbs/fixed_{}.pdb'.format(pdbid)):
        filename = prefix+'MasifOutput/00-raw_pdbs/fixed_{}.pdb'.format(pdbid)
        protein = 'fixed_'+pdbid
    elif os.path.exists(prefix+'MasifOutput/00-raw_pdbs/{}.pdb'.format(pdbid)):
        filename = prefix+'MasifOutput/00-raw_pdbs/{}.pdb'.format(pdbid)
        protein = pdbid
    else:
        print('pdb file not found')
        return None
    cmd.reinitialize()
    cmd.load(filename)
    if removeHs:
        cmd.remove('hydro')
    protein_coords = []
    cmd.iterate_state(-1, "chain "+"+".join([x for x in chains])+" and not het", "protein_coords.append((x,y,z))", space=locals())
    protein_coords = np.array(protein_coords)
    return protein_coords
    
    
def get_masif_coord_normal(pdbid):
    mesh = pymesh.load_mesh(prefix+'MasifOutput/01-benchmark_surfaces/{}.ply'.format(pdbid))
    
    x = mesh.get_vertex_attribute("vertex_x")
    y = mesh.get_vertex_attribute("vertex_y")
    z = mesh.get_vertex_attribute("vertex_z")
    prt_coord = np.hstack([x,y,z])
    mesh.add_attribute("vertex_normal")
    normal = mesh.get_vertex_attribute("vertex_normal")
    return prt_coord, normal

def check_angle(coord, masif, normal):
    cosine_list = np.matmul((coord-masif)[:,None], normal[:,:,None])[:,0,0]
    return cosine_list


def getIndexedFaceSet(wrlname):
    with open(wrlname, 'r') as f:
        for line in f:
            if 'geometry IndexedFaceSet {' in line:
                break
        else:
            print("Could not find IndexedFaceSet")
            return None
        assert 'coord Coordinate' in f.readline()
        assert 'point' in f.readline()
        vertices = []
        for line in f:
            if not line.strip(): continue
            line = line.strip()
            if ']' in line:
                break
            if line.endswith(','):
                line = line[:-1]
            vertices.append([float(item) for item in line.split()])
        assert '}' in f.readline()
        assert 'coordIndex' in f.readline()
        
        for line in f:
            if 'normalPerVertex' in line:
                break
        else:
            print("Could not find normalPerVertex")
            return None
        assert 'normal Normal' in f.readline()
        assert 'vector' in f.readline()
         
        normals = []
        for line in f:
            if not line.strip(): continue
            line = line.strip()
            if ']' in line:
                break
            if line.endswith(','):
                line = line[:-1]
            normals.append([float(item) for item in line.split()[:3]])
    return np.array(vertices), np.array(normals)

def get_mesh_vertices(pdbid, chains):
    if os.path.exists(prefix+'MasifOutput/00-raw_pdbs/fixed_{}.pdb'.format(pdbid)):
        filename = prefix+'MasifOutput/00-raw_pdbs/fixed_{}.pdb'.format(pdbid)
        protein = 'fixed_'+pdbid
    elif os.path.exists(prefix+'MasifOutput/00-raw_pdbs/{}.pdb'.format(pdbid)):
        filename = prefix+'MasifOutput/00-raw_pdbs/{}.pdb'.format(pdbid)
        protein = pdbid
    else:
        print('pdb file not found')
        return None
    
    pymol.cmd.reinitialize()
    pymol.cmd.load(filename)
    pymol.cmd.hide('everything')
     
#     pymol.cmd.set('surface_quality', -2)
    pymol.cmd.set('solvent_radius', 1.5)
    pymol.cmd.show('surface', 'chain {}'.format("+".join(chains)))
    pymol.cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    pymol.cmd.save(tmp_dir + pdbid + '.wrl')
    vertices, normals = getIndexedFaceSet(tmp_dir + pdbid + '.wrl')
#     os.remove('./tmp/' + pdbid + '.wrl')
    return vertices, normals


def sample_coords(pdbid, seed=1234, extend=5, pl_thre=2, pr_thre=5, isprint=False):
    # use masif: pl=0, pr=2.5
    # masif_coords, normal = get_masif_coord_normal(pdbid, datatype)
    protein_exposed = load_protein_coords(*pdbid.split('_'))
    if isprint:
#         print('masif', masif_coords.shape)
        print('protein',protein_exposed.shape)
#         print('normal', normal.shape)
    try:
        pymol_coords, pymol_normal = get_mesh_vertices(*pdbid.split('_'))
        if isprint:
            print('pymol vertices and normals', pymol_coords.shape, pymol_normal.shape)
        pymol_index = np.random.choice(np.arange(len(pymol_coords)), 
                                       int(np.ceil(len(pymol_coords)/10)),
                                       replace=False)
        masif_coords, normal = pymol_coords[pymol_index], pymol_normal[pymol_index]
        if isprint:
            print('pymol vertices and normals sampled', masif_coords.shape, normal.shape)
    except:
        print('trying to use masif normals')
        masif_coords, normal = get_masif_coord_normal(pdbid)
    
    np.random.seed(seed)
    step = 2
    randomness = (np.random.rand(3)*2-1)*step 
    coord_bond_l = protein_exposed.min(axis=0) - extend + randomness
    coord_bond_r = protein_exposed.max(axis=0) + extend + randomness

    n_x, n_y, n_z = int((coord_bond_r[0] - coord_bond_l[0])/step), \
                int((coord_bond_r[1] - coord_bond_l[1])/step), int((coord_bond_r[2] - coord_bond_l[2])/step)
    
    x_coords = np.linspace(coord_bond_l[0], coord_bond_r[0], num = n_x)
    y_coords = np.linspace(coord_bond_l[1], coord_bond_r[1], num = n_y)
    z_coords = np.linspace(coord_bond_l[2], coord_bond_r[2], num = n_z)
    mask = np.ones((n_x, n_y, n_z))
    points = np.where(mask)
    initial_coords = np.array([x_coords[points[0]], y_coords[points[1]], z_coords[points[2]]]).transpose()  
        
    if isprint:
        print('number of initial points', initial_coords.shape)
    
    min_arg, min_dist = pairwise_distances_argmin_min(initial_coords, protein_exposed)
    pp_distance_filter = (min_dist >= pl_thre) & (min_dist <= pr_thre)
    if isprint:
        print('pp_distance_filter', pp_distance_filter.shape, pp_distance_filter.sum())
    
    v1_coords = initial_coords[pp_distance_filter]
    if isprint:
        print('v1_coords', v1_coords.shape)
    min_arg, min_dist = pairwise_distances_argmin_min(v1_coords, masif_coords)
    pm_distance_filter = (min_dist <= 2.5) 
    near_masif_idx = min_arg[pm_distance_filter]
    normal_near = normal[near_masif_idx]
    cosine = check_angle(v1_coords[pm_distance_filter], masif_coords[near_masif_idx], normal_near)
    normal_filter = (cosine>0)
    v2_coords = v1_coords[pm_distance_filter][normal_filter]
    
    return v2_coords

if __name__ == '__main__':
    
    pdbid = sys.argv[1]
    da = int(sys.argv[2]) # default 6
    repeat = 1
    
    outdir = prefix+'AnchorOutput/{}_da_{}/'.format(pdbid, da)
    if os.path.exists(outdir + 'anchors.npy'):
        print('anchor already exists!')
        exit()
    
    t0 = time.time()
    np.random.seed(1234)
    seed_list = np.random.randint(0, 10000, repeat)
    
    repeat_list = []
    for i_rep in range(repeat):
        points = sample_coords(pdbid, seed=seed_list[i_rep], isprint=False)
        ac_complete = AgglomerativeClustering(n_clusters=None, distance_threshold=da, affinity='euclidean', linkage='complete')
        ac_complete.fit(points)
        centers = []
        for ni in range(ac_complete.n_clusters_):
#             if np.sum([ac_complete.labels_==ni]) <= 5:
#                 continue
            centers.append(np.mean(points[ac_complete.labels_==ni], axis=0))
        repeat_list.append(np.array(centers))
    
    t1 = time.time()
    print('anchor generation aug {} shape {} time {} s'.format(repeat, np.array(repeat_list).shape, t1-t0))

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    np.save(outdir + 'anchors.npy', repeat_list)
