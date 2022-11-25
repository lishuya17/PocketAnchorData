### Data processing for “PocketAnchor: Learning Structure-based Pocket Representations for Protein-Ligand Interaction Prediction”

This repository is for reproducing the data processing steps, or anchor generation for customized datasets. 

The training code for this paper can be found at https://github.com/tiantz17/PocketAnchor.

Note: for just reproducing the results in the paper, there is no need for running the data processing steps, as it may take several days. The processed datasets used in our paper will be provided at https://github.com/tiantz17/PocketAnchor very soon.

## For data processing:

### Step 0. Prepare the environment

If generating MaSIF precomputed features is needed, please pull the corresponding docker image for easy calculation of MaSIF precomputed features:

```
docker pull lishuya17/masif-mini-server:20220924
```

And git clone this repository.

Before calculating the MaSIF precomputed features, please start the service:

```
cd PocketAnchorData
mkdir AnchorOutput
mkdir MasifOutput
docker run -it -p 1213:1213 -v $PWD/input/:/input/ -v $PWD/MasifOutput/:/masif/data/masif_ligand/data_preparation/ lishuya17/masif-mini-server:20220924
```

Run the following commands in the started container, and adjust the num_workers (default: 16) according to your computational resource. Note that one worker stands for one MaSIF process, which may approximately use 1-6 CPU cores.

```
cd /masif/data/masif_ligand/
nohup python -u data_prepare_all_PDB.py --num_workers 16 > prepare_all.log &
nohup python -u masif_server.py > server.log &

cd /root/
nohup python -u clean_tmp.py &
```

Then, follow the steps in jupyter notebook files to generate inputs of the PocketAnchor method.

Environment:

```
scikit-learn
pymol
scipy
```

### Step1. Obtain the MaSIF precomputed features:

Just load the data and query the MaSIF server docker. Examples are provided in Step1.Prepare_pdb_files_and_MaSIF_precomputation.ipynb

The server may process hundreds of samples in an hour, depends on the defined num_workers. The status of the queries can be checked, and the return values can be the following types:

- Pending: waiting for doing other tasks
- Doing: calculating masif features for this task
- Done: found existing masif features for this task
- Failed: finished calculation but did not generate result (Re-submit with redo=True or fixing the pdb file may help sometimes)
- No such task: no feature files and no state record for this task

### Step2. Calculate the anchor features:

Please follow the examples are provided in Step2.Obtain_anchors.ipynb. All the required source code can be found in /src.

### Step3. Prepare model inputs:

For each task, run the corresponding Step3.X

This step will organize the generated features in /AnchorOutput to produce more compact files.
