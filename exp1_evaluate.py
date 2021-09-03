import pickle
import numpy as np
from sklearn.metrics import pairwise_distances
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
import time


num_query = 1000
save_output_dir = Path("./outputs_exp1/")
hidden_size = 128
batch_size = 128
bidirectional = 0
n_layers = 1
freeze = 1
#features_type = "predictor"
#features_type = "middle"
features_type = "encoder"
#label_path = "../t2vec-master/experiment/exp1/exp1-trj.label"
label_path = "../didi_new/experiment/exp1/exp1-trj.label"
#city = "chengdu"
#city = "porto"
city = "chengdu"

def ranksearch_true(query, query_label, db, db_label):
    ranks = []
    precisions = 0
    # len_q, len_db
    dis_matrix = pairwise_distances(query, db, n_jobs=1)
    idxs = np.argsort(np.argsort(dis_matrix, -1), -1)
    for i in range(idxs.shape[0]):
        for j in range(idxs.shape[1]):
            if query_label[i] == db_label[j]:
                ranks.append(idxs[i][j])
                if idxs[i][j] == 0:
                    precisions += 1
                break
    return ranks, precisions / len(query_label)

def ranksearch_true_ckdtree(query, query_label, db, db_label):
    ranks = []
    precisions = 0
    kdtree = cKDTree(db)
    max_len = len(db)

    def ranksearch_subprocedure(x, xlabel):
        k = 5
        while k < max_len:
            _, idxs = kdtree.query(x, k, n_jobs=8)
            for j in range(len(idxs)):
                if db_label[idxs][j] == xlabel:
                    return j
            k = k * 2

    # len_q, len_db
    for i in range(len(query)):
        ranks.append(ranksearch_subprocedure(query[i], query_label[i]))

    return ranks, precisions / len(query_label)

def read_labels(filepath):
    labels = []
    with open(filepath) as f:
        for label in f:
            labels.append(int(label.strip()))
    return np.array(labels)

epochs_list = []
db_size_list = []
ranks_list = []
precision_list = []
for save_output_path in save_output_dir.iterdir():
    stem = save_output_path.stem.split("_")
    #if not "lr" in stem:
    #    continue
    if not city in save_output_path.stem:
        continue
    if not (stem[0] == features_type and int(stem[3]) == hidden_size and int(stem[5]) == batch_size and int(stem[7]) == bidirectional
        and int(stem[9]) == n_layers and int(stem[11]) == freeze):
        continue
    if not (int(stem[1]) == 39):
        continue
    print()
    print(save_output_path)

    vecs = pickle.load(open(save_output_path, "rb"))
    labels = read_labels(label_path)

    query, db = vecs[:num_query], vecs[num_query:]
    query_label, db_label = labels[:num_query], labels[num_query:]

    dbsizes = [20000, 40000, 60000, 80000, 100000]
    #dbsizes = [100000]
    for dbsize in dbsizes:
        #ranks, precision = ranksearch_true_ckdtree(query, query_label, db[:dbsize], db_label[:dbsize])
        start_time = time.time()
        ranks, precision = ranksearch_true(query, query_label, db[:dbsize], db_label[:dbsize])
        print("mean rank:", np.mean(ranks), "with dbsize:", dbsize)
        print(time.time()-start_time)

        epochs_list.append(int(stem[1]))
        db_size_list.append(dbsize)
        ranks_list.append(np.mean(ranks))
        precision_list.append(precision)

pd.DataFrame({"epochs": epochs_list, "dbsize": db_size_list, "ranks": ranks_list, "precision": precision_list}).to_csv(
        "batch_results/{}_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}_{}.csv".format(
        features_type, hidden_size, batch_size, bidirectional, n_layers, freeze, city), index=False)
