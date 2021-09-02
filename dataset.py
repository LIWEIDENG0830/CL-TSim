import torch
from pathlib import Path
from torch.utils.data import DataLoader
import time
import numpy as np


class Dataset:
    def __init__(self, src_file_path, trg_file_path, max_len, n_views):
        self.src_file_path = src_file_path
        self.trg_file_path = trg_file_path
        self.max_len = max_len
        self.n_views = n_views
        self.src_data, self.src_data_len = self.read_file(src_file_path)
        self.trg_data, self.trg_data_len = self.read_file(trg_file_path)

    def read_file(self, filepath):
        start_time = time.time()
        print("Reading file: {}".format(filepath))
        data_list = []
        data_len = []
        with open(filepath) as f:
            multi_views = []
            multi_views_len = []
            for idx, traj in enumerate(f):
                if idx%self.n_views == 0:
                    if multi_views != []:
                        # delete the identity map
                        multi_views = multi_views[1:]
                        multi_views_len = multi_views_len[1:]
                        data_list.append(multi_views)
                        data_len.append(multi_views_len)
                    multi_views = []
                    multi_views_len = []
                traj = [int(point) for point in traj.strip().split(" ")]
                if len(traj) > self.max_len:
                    traj = traj[:self.max_len]
                    traj_len = self.max_len
                else:
                    traj_len = len(traj)
                    traj = traj + [0]*(self.max_len-traj_len)
                multi_views.append(traj)
                multi_views_len.append(traj_len)
        print("Reading cost: {:.4f}".format(time.time()-start_time))
        return data_list, data_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, i):
        # randomly choose a view of trg_data
        j = np.random.randint(len(self.src_data[i]))
        return torch.tensor([self.src_data[i][j]] + [self.trg_data[i][j]], dtype=torch.long), \
               torch.tensor([self.src_data_len[i][j]] + [self.trg_data_len[i][j]], dtype=torch.long)


def get_dataloader(dataset_dir, max_len, batch_size, n_views=20, shuffle=True, num_workers=4):

    train_src_path = Path(dataset_dir) / "train.src"
    train_trg_path = Path(dataset_dir) / "train.trg"
    #train_src_path = Path(dataset_dir) / "val.src"
    #train_trg_path = Path(dataset_dir) / "val.trg"
    val_src_path = Path(dataset_dir) / "val.src"
    val_trg_path = Path(dataset_dir) / "val.trg"

    train_dataset = Dataset(train_src_path, train_trg_path, max_len, n_views)
    val_dataset = Dataset(val_src_path, val_trg_path, max_len, n_views)
    import pdb; pdb.set_trace()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader

