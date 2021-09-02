from pathlib import Path
from gensim.models import word2vec
import time


epochs = 5
size = 128
gridsize = 100
dataset_dir = Path("../t2vec-master/data/porto_{}/".format(gridsize))
vec_dir = Path("./pretrained/")
if not vec_dir.exists():
    vec_dir.mkdir()
vec_path = vec_dir / "porto_{}_gridsize_{}.txt".format(size, gridsize)


def read_file(filepath):
    trajs = []
    with open(filepath) as f:
        for traj in f:
            traj = [str(point) for point in traj.strip().split(" ")]
            trajs.append(traj)
    return trajs

train_trg_path = dataset_dir / "train.trg"
val_trg_path = dataset_dir / "val.trg"

start_time = time.time()
trajs = read_file(train_trg_path)
print("Loading {}, cost time: {:.4f}".format(train_trg_path, time.time()-start_time))
start_time = time.time()
trajs.extend(read_file(val_trg_path))
print("Loading {}, cost time: {:.4f}".format(val_trg_path, time.time()-start_time))

start_time = time.time()
model = word2vec.Word2Vec(trajs, min_count=1, size=size, workers=16, sg=1, negative=20,
        alpha=0.025, iter=epochs, compute_loss=True)

model.wv.save_word2vec_format(vec_path, binary=False)
print("Training time: {:.4f}".format(time.time()-start_time))
