import torch
from models import LSTMSimCLR
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm


max_len = 100
hidden_size = 128
batch_size = 128
bidirectional = 0
n_layers = 1
freeze = 1
max_vocab_size = 18866
device = "cuda:0"
features_type = "encoder"

#test_model_dir = "./log/1629299813.656259/"
test_model_dir = "./log/1629475290.7133796/"
test_model_type = "LSTM"
save_output_format = "./outputs_exp3/{}_r{}_{}_{}_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}_chengdu.pkl"

for aug_type in ["distorting", "downsampling"]:
    for ratio in [2, 3, 4, 5, 6]:
        test_data_path = "../t2vec-master/experiment/exp3/{}-r{}-trj.t".format(aug_type, ratio)

        class Dataset:
            def __init__(self, filepath, max_len):
                self.filepath = filepath
                self.max_len = max_len
                self.trajs, self.trajs_len =self.read_data(filepath, max_len)

            def read_data(self, filepath, max_len):
                trajs = []
                trajs_len = []
                with open(filepath) as f:
                    for traj in f:
                        traj = [int(point) for point in traj.strip().split(" ")]
                        if len(traj) > max_len:
                            traj = traj[:max_len]
                            traj_len = max_len
                        else:
                            traj_len = len(traj)
                            traj = traj + [0] * (max_len-traj_len)
                        trajs_len.append(traj_len)
                        trajs.append(traj)
                return trajs, trajs_len

            def __len__(self):
                return len(self.trajs)

            def __getitem__(self, i):
                return torch.tensor(self.trajs[i], dtype=torch.long), \
                       torch.tensor(self.trajs_len[i], dtype=torch.long)


        device = torch.device(device)
        for test_model_path in Path(test_model_dir).iterdir():
            if not test_model_path.name.startswith("checkpoint_"):
                continue
            if test_model_path.name != "checkpoint_0039_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}.pth.tar".format(
                    hidden_size, batch_size, bidirectional, n_layers, freeze):
                continue
            epoch = test_model_path.stem.split("_")[1]
            save_output_path = save_output_format.format(aug_type, ratio, features_type, epoch, hidden_size, batch_size, bidirectional, n_layers, freeze)
            print("test_model_path", test_model_path)
            print("save_output_path", save_output_path)

            checkpoint = torch.load(test_model_path, map_location=device)
            state_dict = checkpoint["state_dict"]

            if test_model_type == "LSTM":
                model = LSTMSimCLR(max_vocab_size, hidden_size, bidirectional, n_layers)
                model.load_state_dict(state_dict)
                model.to(device)
            else:
                raise ValueError("Unknown model type!")

            test_dataset = Dataset(test_data_path, max_len)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            features_list = []
            with torch.no_grad():
                model.eval()
                for batch in tqdm(test_dataloader):
                    trajs, trajs_len = [b.to(device) for b in batch]
                    # batch_size, hidden_size * n_direction
                    if features_type == "encoder":
                        features = model.encode_by_encoder(trajs, trajs_len)
                    else:
                        raise ValueError
                    features_list.append(features.cpu())
                features_list = torch.cat(features_list, dim=0).numpy()

            pickle.dump(features_list, open(save_output_path, "wb"))
