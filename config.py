from dataclasses import dataclass

@dataclass
class Config:

    datadir: str = "../t2vec-master/data/porto_100_org/"
    cell_embedding: str = "./pretrained/porto_128.txt"
    freeze: int = 1
    n_layers: int = 1
    bidirectional: int = 0
    hidden_size: int = 128
    max_len: int = 100
    batch_size: int = 128
    n_views: int = 20
    temperature: float = 0.07
    epochs: int = 40
    gpu_id: int = 2
    arch: str = "LSTM"
    max_vocab_size: int = 18866
    lr: float = 0.0001
    weight_decay: float = 1e-4
    logdir: str = "log"
    log_every_n_steps: int = 100
