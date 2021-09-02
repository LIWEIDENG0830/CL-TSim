import torch
from transformers import HfArgumentParser
from config import Config
import dataset
from models import LSTMSimCLR, BaseSimCLR
from utils import load_pretrained

def main():
    parser = HfArgumentParser(Config)
    args = parser.parse_args_into_dataclasses()[0]

    train_dataloader, val_dataloaer = dataset.get_dataloader(args.datadir, args.max_len, args.batch_size)

    pretrained_cell_embeddings = load_pretrained(args.cell_embedding, args.max_vocab_size)
    if args.arch == "LSTM":
        model = LSTMSimCLR(args.max_vocab_size, args.hidden_size, args.bidirectional, args.n_layers)
        if args.freeze == 3:
            for parm in model.embedding.parameters():
                parm.requires_grad = False
            print("Pretrain-Free-and-Freeze")
        else:
            model.load_pretrained_embedding(pretrained_cell_embeddings, args.freeze)
        model.to(torch.device("cuda:{}".format(args.gpu_id)))
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0,
                                                           last_epoch=-1)

    simclr = BaseSimCLR(model=model, args=args, optimizer=optimizer, scheduler=scheduler)
    simclr.train(train_dataloader, val_dataloaer)


if __name__ == '__main__':
    main()

