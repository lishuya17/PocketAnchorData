### Data processing for “PocketAnchor: Learning Structure-based Pocket Representations for Protein-Ligand Interaction Prediction”


If generating MaSIF precomputed features is needed, please download the corresponding docker container for easy calculation of MaSIF precomputed features:

```
docker pull lishuya17/masif-mini-server:tagname
```

Before calculating the MaSIF precomputed features, please start the service:

```
cd PocketAnchorData
mkdir AnchorOutput
mkdir MasifOutput
docker run -it -p 1213:1213 -v $PWD/input/:/input/ -v $PWD/MasifOutput/:/masif/data/masif_ligand/data_preparation/ lishuya17/masif-mini-server:20220924

cd /masif/data/masif_ligand/
nohup python -u data_prepare_all_PDB.py --num_workers 16 > prepare_all.log &
nohup python -u masif_server.py > server.log &

cd /root/
nohup python -u clean_tmp.py &
```

Then, follow the steps in jupyter notebook files to generate inputs of the PocketAnchor method.
