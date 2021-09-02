# CL-TSim
This repo is about paper "Efficient Trajectory Similarity Computation with Contrastive Learning".

# Pre-processing
We follow t2vec (https://github.com/boathit/t2vec) to do the pre-processing.

Specifically, you should download the codes of t2vec firstly. After that, you should change the working directory to t2vec. Then for the Porto dataset, you can do as follows.
```
$ curl http://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip -o data/porto.csv.zip
$ unzip data/porto.csv.zip
$ mv train.csv data/porto.csv
$ cd preprocessing
$ julia porto2h5.jl
$ julia preprocess.jl
```
After pre-processing, you can get the training data (i.e., train.trg and train.src) in the data directory of t2vec. 

# Grid Pre-training
To pre-train the representations of grids, you should run "grids_pretraining.py" in this repo. 

P.S. you should change the value of variables (i.e., dataset_dir and vec_dir) in "grids_pretraining.py", in which dataset_dir is the directory of training data and vec_dir is to saving trained representations.

# Model Training
To train CL-TSim, you should change the value of variable (i.e., datadir and cell_embedding) in "config.py", where datadir is the directory of training data and cell_embedding is the path of pre-trained representations. Then you can train CL-TSim as follows.
```
python main.py
```
After training is done (around 1 hour by using GeForce GTX 1080 Ti), you can see the trained model in "log" directory.

# Reproducibility
To reproduce the results stated in our paper, you can apply the following steps.
## Self-Similarity and Cross-Similarity data generation
Similar to Pre-processing, we follow t2vec to generate data for self-similairty and cross-similairty.
