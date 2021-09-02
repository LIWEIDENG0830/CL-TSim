import pickle
from pathlib import Path
import numpy as np
import pandas as pd


#aug_type = "downsampling"
#aug_type = "distorting"
num_query = 1000
hidden_size = 128
batch_size = 128
bidirectional = 0
n_layers = 1
freeze = 1
features_type = "encoder"

# downsampling_r2_encoder_0039_hiddensize_128_batchsize_128_bidirectional_1_nlayers_1_freeze_1.pkl
for aug_type in ["distorting", "downsampling"]:
    for ratio in [2, 3, 4, 5, 6]:
        save_output_dir = Path("./outputs_exp3/")

        CDD_list = []
        epochs_list = []

        for save_output_path in save_output_dir.iterdir():
            stem = save_output_path.stem.split("_")
            if not "new" in save_output_path.stem:
                continue
            if not (stem[0] == aug_type and int(stem[1][-1]) == ratio and stem[2] == features_type and int(stem[5]) == hidden_size and
                    int(stem[7]) == batch_size and int(stem[9]) == bidirectional and int(stem[11]) == n_layers and int(stem[13]) == freeze):
                continue

            print()
            print(save_output_path)

            vecs = pickle.load(open(save_output_path, "rb"))
            # num_query, hidden_size; num_query, hidden_size
            query, db = vecs[:10000], vecs[10000:]
            query = query[:num_query]
            db = db[:num_query]

            # num_query, num_query
            query_mul = np.dot(query, query.transpose())
            #query_mul = query_mul * (1 - np.identity(query_mul.shape[0]))
            # num_query, num_query
            db_mul = np.dot(db, db.transpose())
            #db_mul = db_mul * (1 - np.identity(db_mul.shape[0]))

            CDD = np.mean(np.abs((query_mul - db_mul)) / db_mul)
            print("CDD", CDD)

            CDD_list.append(CDD)
            epochs_list.append(int(stem[3]))

        pd.DataFrame({"epochs": epochs_list, "CDD": CDD_list}).to_csv(
                "exp3_results/{}_r{}_{}_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}_new.csv".format(
                aug_type, ratio, features_type, hidden_size, batch_size, bidirectional, n_layers, freeze), index=False)
