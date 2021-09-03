import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def ranksearch_true(query, query_label, db, db_label):
    ranks = []
    precisions = 0
    # len_q, len_db
    dis_matrix = pairwise_distances(query, db, n_jobs=8)
    idxs = np.argsort(np.argsort(dis_matrix, -1), -1)
    for i in range(idxs.shape[0]):
        for j in range(idxs.shape[1]):
            if query_label[i] == db_label[j]:
                ranks.append(idxs[i][j])
                if idxs[i][j] == 0:
                    precisions += 1
                break
    return ranks, precisions / len(query_label)

def read_labels(filepath):
    labels = []
    with open(filepath) as f:
        for label in f:
            labels.append(int(label.strip()))
    return np.array(labels)

#aug_type = "downsampling"
#aug_type = "distorting"
num_query = 1000
db_size = 20000
hidden_size = 128
batch_size = 128
bidirectional = 0
n_layers = 1
freeze = 1
features_type = "encoder"
#label_path_format = "../t2vec-master/experiment/exp2/{}-r{}-trj.label"
label_path_format = "../didi_200/experiment/exp2/{}-r{}-trj.label"
city = "chengdu"

# downsampling_r2_encoder_0039_hiddensize_128_batchsize_128_bidirectional_1_nlayers_1_freeze_1.pkl
for aug_type in ["distorting", "downsampling"]:
    for ratio in [2, 3, 4, 5, 6]:
        save_output_dir = Path("./outputs_exp2/")

        ranks_list = []
        precision_list = []
        epochs_list = []
        db_size_list = []

        for save_output_path in save_output_dir.iterdir():
            if not city in save_output_path.stem:
                continue
            stem = save_output_path.stem.split("_")
            if not (stem[0] == aug_type and int(stem[1][-1]) == ratio and stem[2] == features_type and int(stem[5]) == hidden_size and
                    int(stem[7]) == batch_size and int(stem[9]) == bidirectional and int(stem[11]) == n_layers and int(stem[13]) == freeze):
                continue

            print()
            print(save_output_path)

            vecs = pickle.load(open(save_output_path, "rb"))
            labels = read_labels(label_path_format.format(aug_type, ratio))
            # num_query, hidden_size; num_query, hidden_size
            query, db = vecs[:1000], vecs[1000:]
            query_label, db_label = labels[:1000], labels[1000:]
            query = query[:num_query]
            db = db[:db_size]

            # num_query, num_db
            ranks, precision = ranksearch_true(query, query_label, db, db_label)
            print("ranks", np.mean(ranks), "precision", precision)

            ranks_list.append(np.mean(ranks))
            precision_list.append(precision)
            epochs_list.append(int(stem[3]))
            db_size_list.append(db_size)

        pd.DataFrame(
            {"epochs": epochs_list, "dbsize": db_size_list, "ranks": ranks_list, "precision": precision_list}).to_csv(
            "exp2_results/{}_r{}_{}_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}_{}.csv".format(
                aug_type, ratio, features_type, hidden_size, batch_size, bidirectional, n_layers, freeze, city), index=False)
