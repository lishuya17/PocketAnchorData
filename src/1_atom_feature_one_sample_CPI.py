import sys, os, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymol import cmd
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from collections import defaultdict

prefix = './'

neighbor_table = {
    'GLY': {'N': ['CA'], 'CA':['N', 'C'], 'C': ['CA', 'O'], 'O':['C']},
    'ALA': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA']},
    'PRO': {'N': ['CA', 'CD'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD'], 'CD':['N', 'CG']},
    'VAL': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG1', 'CG2'], 
            'CG1':['CB'], 'CG2':['CB']},
    'LEU': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD1', 'CD2'], 'CD1':['CG'], 'CD2':['CG']},
    'ILE': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG1', 'CG2'], 
            'CG1':['CB', 'CD1'], 'CG2':['CB'], 'CD1':['CG1']},
    'MET': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'SD'], 'SD':['CG', 'CE'], 'CE':['SD']},
    
    'PHE': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD1', 'CD2'], 'CD1':['CG', 'CE1'], 'CD2':['CG', 'CE2'],
            'CE1':['CD1', 'CZ'], 'CE2':['CD2', 'CZ'], 'CZ':['CE1', 'CE2']},
    'TYR': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD1', 'CD2'], 'CD1':['CG', 'CE1'], 'CD2':['CG', 'CE2'],
            'CE1':['CD1', 'CZ'], 'CE2':['CD2', 'CZ'], 'CZ':['CE1', 'CE2', 'OH'], 'OH':['CZ']},
    'TRP': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD1', 'CD2'], 'CD1':['CG', 'NE1'], 'CD2':['CG', 'CE2', 'CE3'],
            'NE1':['CD1', 'CE2'], 'CE2':['NE1', 'CD2', 'CZ2'], 'CE3':['CD2', 'CZ3'],
            'CZ2':['CE2', 'CH2'], 'CZ3':['CE3', 'CH2'], 'CH2':['CZ2', 'CZ3']},
    
    'SER': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'OG'], 'OG':['CB']},
    'THR': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'OG1', 'CG2'], 
            'CG2':['CB'], 'OG1':['CB']},
    'CYS': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'SG'], 'SG':['CB']},
    'ASN': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'OD1', 'ND2'], 'OD1':['CG'], 'ND2':['CG']},
    'GLN': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD'], 'CD':['CG', 'NE2', 'OE1'], 'OE1':['CD'], 'NE2':['CD']},
    
    'LYS': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD'], 'CD':['CG', 'CE'], 'CE':['CD', 'NZ'], 'NZ':['CE']},
    'HIS': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CG'], 'CG':['CB', 'ND1', 'CD2'], 'ND1':['CG', 'CE1'], 'CD2':['CG', 'NE2'],
            'CE1':['ND1', 'NE2'], 'NE2':['CE1', 'CD2']},
    'ARG': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD'], 'CD':['CG', 'NE'], 'NE':['CD', 'CZ'], 'CZ':['NE', 'NH1', 'NH2'],
            'NH1':['CZ'], 'NH2':['CZ']},
    
    'ASP': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'],
            'CG':['CB', 'OD1', 'OD2'], 'OD1':['CG'], 'OD2':['CG']},
    'GLU': {'N': ['CA'], 'CA':['N', 'C', 'CB'], 'C': ['CA', 'O'], 'O':['C'], 'CB':['CA', 'CG'], 
            'CG':['CB', 'CD'], 'CD':['CG', 'OE1', 'OE2'], 'OE1':['CD'], 'OE2':['CD']}  
}

for key in neighbor_table:
    neighbor_table[key]['OXT'] = ['C']

    
    
ELEM_LIST = ['C', 'O', 'N', 'S', 'P', 'ION']
ION_LIST = ['ZN', 'MN', 'CA','MG', 'K', 'CU', 'NI', 'FE', 'SE', 'CD', 'NA', 'CO', 'HG', 'SR', 'CS', 'GA', 'RB', 'IN', 'Cl', 'Du', \
            'Zn', 'Co', 'Mg', 'LI', 'Cd', 'Ca', 'AU', 'Mn', 'CL', 'X', 'AS', 'B', 'F', 'I']
AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', \
           'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HETATM']  # HET: ions
SS_LIST = ['', 'H', 'L', 'S']
AA_TO_ATOM = {'GLY': ['OXT', 'C', 'N', 'O', 'CA'],
              'LEU': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2'],
              'ALA': ['OXT', 'C', 'N', 'O', 'CA', 'CB'],
              'HIS': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'ND1', 'NE2'],
              'PHE': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
              'TRP': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CH2', 'CZ2', 'CZ3', 'NE1'],
              'TYR': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
              'ASN': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'ND2', 'OD1'],
              'VAL': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG1', 'CG2'],
              'GLU': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
              'SER': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'OG'],
              'ASP': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'OD1', 'OD2'],
              'PRO': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD'],
              'MET': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CE', 'SD'],
              'ILE': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG1', 'CG2', 'CD1'],
              'THR': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG2', 'OG1'],
              'LYS': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ'],
              'ARG': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'CZ', 'NE', 'NH1', 'NH2'], 
              'GLN': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'NE2', 'OE1'], 
              'CYS': ['OXT', 'C', 'N', 'O', 'CA', 'CB', 'SG'],
              'HETATM': ['X']}

SHARED_ATOMS = ['OXT', 'C', 'N', 'O', 'CA']
AA_ATOM_LIST = []
for aa, atom_list in AA_TO_ATOM.items():
    for atom in atom_list:
        if atom not in SHARED_ATOMS:
            AA_ATOM_LIST.append(aa+'_'+atom)
AA_ATOM_LIST = SHARED_ATOMS + AA_ATOM_LIST



def guess_neighbor_name(chain, resi, resn, name):
    nei_list = []
    for n in neighbor_table[resn][name]:
        nei_list.append('{}_{}_{}'.format(chain, resi, n))
    if name == 'N':
        nei_list.append('{}_{}_{}'.format(chain, resi-1, 'C'))
    if name == 'C':
        nei_list.append('{}_{}_{}'.format(chain, resi+1, 'N'))
    return nei_list



def get_pocket_info_and_coord(pdbid, anchor_coord):
    t1 = time.time()
    cmd.reinitialize()
    cmd.load(prefix+'MasifOutput/01-benchmark_pdbs/{}.pdb'.format(pdbid))
    cmd.remove('hydro')
    selName = cmd.get_unused_name("sele_")
    cmd.select(selName, "all".format(pdbid))
    
    pocket_info = []
    pocket_coord = []
    cmd.iterate(selName, "pocket_info.append([chain, resv, resn, type, name, ID, elem, b, formal_charge, partial_charge, vdw, protons, ss, geom, valence])", space=locals())
    cmd.iterate_state(-1, selName, "pocket_coord.append((x,y,z))", space=locals())
    pocket_coord = np.array(pocket_coord)
    assert len(pocket_info) == pocket_coord.shape[0], pdbid
    
    mt_arg, min_mt = pairwise_distances_argmin_min(pocket_coord, anchor_coord)
    
    resi_set = set()
    for i, item in enumerate(pocket_info):
        if item[6] == 'H':
            continue
        chain = item[0]
        resi = str(item[1])
        if min_mt[i] <= 6:  # 6A: max non-covalent bond distance
            resi_set.add(chain+resi)
    atom_idx = []
    for i in range(len(pocket_info)):
        if pocket_info[i][0]+str(pocket_info[i][1]) in resi_set and pocket_info[i][6] != 'H':
            atom_idx.append(i)
    pocket_info2 = [pocket_info[i] for i in atom_idx]

    pocket_coord = pocket_coord[atom_idx]
    
    info_to_ID = {}
    for i, item in enumerate(pocket_info2):
        chain, resi, resn, type_, name, ID = item[:6]
        info_to_ID['{}_{}_{}'.format(chain, resi, name)] = ID

    for i, item in enumerate(pocket_info2):
        chain, resi, resn, type_, name, ID = item[:6]
        if resn in neighbor_table:
            nei_list = guess_neighbor_name(chain, resi, resn, name)
            nei = [info_to_ID[x] for x in nei_list if x in info_to_ID]
        else:
            selName = cmd.get_unused_name("nei_")
            cmd.select(selName, " neighbor ///{}/{}`{}/{}".format(chain, resn, resi, name))
            nei_pre = []
            cmd.iterate(selName, "nei_pre.append([ID, chain, resv])", space=locals())
            nei = []
            for n1, n2, n3 in nei_pre:
                if n2+str(n3) in resi_set:
                    nei.append(n1)
        pocket_info2[i].append(nei) 

    return pocket_info2, pocket_coord


max_aa_atom = max([len(item) for item in AA_TO_ATOM.values()])
print('max_aa_atom', max_aa_atom)

def get_atom_in_aa(atom, aa):
    if aa == 'HETATM':
        return np.zeros(max_aa_atom).tolist()
    assert aa in AA_TO_ATOM, aa
    res = np.zeros(max_aa_atom)
    try:
        res[AA_TO_ATOM[aa].index(atom)] = 1
    except:
        print(aa, atom)
        xxx
    return res.tolist()

def get_one_hot(item, item_list):
    assert item in item_list, (item, item_list)
    res = np.zeros(len(item_list))
    res[item_list.index(item)] = 1
    return res.tolist()

def get_one_hot_atom(item, item_list):
    if item not in item_list:
        # print('invalid atom', item)
        item = 'HETATM_X'
    res = np.zeros(len(item_list))
    res[item_list.index(item)] = 1
    return res.tolist()

def get_one_hot_aa(item, item_list):
    if item not in item_list:
        # print('invalid aa', item)
        item = 'HETATM'
    res = np.zeros(len(item_list))
    res[item_list.index(item)] = 1
    return res.tolist()

def get_feature(one_info):
    chain, resv, resn, type_, name, index, elem, b, formal_charge, partial_charge, vdw, protons, ss, geom, valence, nei = one_info
    if resn == 'HOH':
        return None
    if elem == 'H':
        return None
    if elem == 'D':
        return None
    atom_type = int(type_ == 'ATOM')
    if not atom_type:
        resn = type_
    if elem in ION_LIST:
        elem = 'ION'
    atom_elem = get_one_hot(elem, ELEM_LIST)
    aa = get_one_hot_aa(resn, AA_LIST)
    if elem == 'ION': 
        atom_in_aa = np.zeros(len(AA_ATOM_LIST)).tolist()
    else:
        if name not in SHARED_ATOMS:
            name = 'HETATM_X' if resn == 'HETATM' else resn+'_'+name
        atom_in_aa = get_one_hot_atom(name, AA_ATOM_LIST)
    ss_type = get_one_hot(ss, SS_LIST)
    feature = atom_elem + aa + atom_in_aa + ss_type + [atom_type, b, formal_charge, vdw, protons, geom, valence]
    return feature

def get_nei_list(index_list, nei_list):
    id_map = {i_old: i_new for i_new, i_old in enumerate(index_list)}
    result = []
    for nei in nei_list:
        res = []
        for i in nei:
            if i in id_map:
                res.append(id_map[i])
        result.append(res)
    return result

def get_feature_bulk(info_list, coord):
    all_features, all_coord, index_list, nei_list = [], [], [], []
    for i, info in enumerate(info_list):
        feature = get_feature(info)
        if feature is not None:
            all_features.append(feature)
            all_coord.append(coord[i])
            index_list.append(info[5])
            nei_list.append(info[-1])
    nei_list = get_nei_list(index_list, nei_list)
    return np.array(all_features), np.array(all_coord), nei_list



if __name__ == '__main__':
    
    pdbid = sys.argv[1]
    
    outdir = prefix+'AnchorOutput/{}_center_{}_da_{}/'.format(pdbid, sys.argv[2], sys.argv[3])
    
    if os.path.exists(outdir+'atom_feature.pk'.format(pdbid)):
        print('atom feature already exists!')
        exit()
    
    anchor_list = np.load(outdir+'anchors.npy', allow_pickle=True)
    
    feature_dict = {}
    for i_rep, anchor in enumerate(anchor_list):
        
        print('repeat', i_rep)
        t1 = time.time()
        info, coord = get_pocket_info_and_coord(pdbid, anchor)
        t2 = time.time()
        print('atom feature raw', pdbid, 'time', t2-t1, 's')

        feature, new_coord, nei_list = get_feature_bulk(info, coord)
        feature_dict[i_rep] = feature, new_coord, nei_list
        

    with open(outdir+'atom_feature.pk'.format(pdbid), 'wb') as f:
        pickle.dump(feature_dict, f)
    t3 = time.time()
    print('atom feature', pdbid, 'time', t3-t2, 's')

    
    



