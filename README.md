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
To pre-train the representations of grids, you should run "grid_pretraining.py" in this repo.
