import time
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import accuracy, save_checkpoint
from pathlib import Path
import json
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional, n_layers, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                            bidirectional=bidirectional, num_layers=n_layers)

    def forward(self, trajs_hidden, trajs_len):
        # trajs_hidden: batch_size * 2 * n_views, seq_len, hidden_size
        # trajs_len: batch_size * 2 * n_views
        #packed_trajs_hidden = pack_padded_sequence(trajs_hidden, trajs_len.detach().cpu(), batch_first=True, enforce_sorted=False)
        # hn: num_layers * n_direction, batch_size * 2 * n_views, hidden_size
        #_, (hn, _) = self.lstm(packed_trajs_hidden)
        #outputs, _ = self.lstm(packed_trajs_hidden)
        outputs, _ = self.lstm(trajs_hidden)
        #hn = hn.transpose(0, 1).reshape(trajs_hidden.shape[0], -1)
        # outputs: batch_size * 2 * n_views, seq_len, hidden_size * n_direction
        # outputs, _ = self.lstm(trajs_hidden)
        # hn: batch_size * 2 * n_views, hidden_size * n_direction
        hn = outputs[torch.arange(trajs_hidden.shape[0]), trajs_len-1]
        #unpacked_output, hn = pad_packed_sequence(packed_output, batch_first=True)
        #return hn.transpose(0, 1).reshape(trajs_hidden.shape[0], -1)
        return hn


class LSTMSimCLR(nn.Module):

    def __init__(self, vocab_size, hidden_size, bidirectional, n_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional = True if bidirectional else False
        self.n_direction = 2 if self.bidirectional else 1
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.encoder = LSTMEncoder(self.hidden_size, self.hidden_size, self.bidirectional, self.n_layers)
        self.predictor = nn.Sequential(
            nn.Linear(self.n_direction*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def load_pretrained_embedding(self, embedding_matrix, freeze):
        # other number of freeze means do not load the pretraining embeddings
        if freeze not in {0, 1}:
            print("No pretraining embeddings")
            return
        freeze = True if freeze else False
        self.embedding = self.embedding.from_pretrained(embedding_matrix, freeze=freeze)

    def forward(self, trajs, trajs_len):
        # trajs: batch_size * 2 , seq_len
        # trajs_len: batch_size * 2

        # trajs_hidden: batch_size * 2, seq_len, hidden_size
        trajs_hidden = self.embedding(trajs)
        # features: batch_size * 2, hidden_size * n_direction
        features = self.encoder(trajs_hidden, trajs_len)
        # features: batch_size * 2, hidden_size
        features = self.predictor(features)

        return features

    def encode_by_encoder(self, trajs, trajs_len):
        trajs_hidden = self.embedding(trajs)
        features = self.encoder(trajs_hidden, trajs_len)
        return features

    def encode_by_predictor(self, trajs, trajs_len):
        return self.forward(trajs, trajs_len)

    def encode_by_middle_layer(self, trajs, trajs_len):
        trajs_hidden = self.embedding(trajs)
        features = self.encoder(trajs_hidden, trajs_len)
        for module in self.predictor:
            features = module(features)
            break
        return features


class BaseSimCLR(object):

    def __init__(self, model, args, optimizer, scheduler):
        super().__init__()

        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

        args.logdir = args.logdir + "/" + str(time.time())
        self.writer = SummaryWriter(logdir=args.logdir)

        config_path = Path(args.logdir) / "config.json"
        with open(config_path, "w") as fconfig:
            json.dump(vars(args), fconfig, indent=4)

        logging.basicConfig(filename=os.path.join(args.logdir,
            'training_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}.log'.format(args.
            hidden_size, args.batch_size, args.bidirectional, args.n_layers, args.freeze)), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(torch.device("cuda:{}".format(self.args.gpu_id)))

    def info_nce_loss(self, features, batch_size):

        labels = torch.cat([torch.tensor([i]*2) for i in range(batch_size)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, val_dataloader=None):

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.gpu_id}.")

        device = torch.device("cuda:{}".format(self.args.gpu_id))
        for epoch_counter in range(self.args.epochs):
            self.model.train()
            for batch in tqdm(train_loader):

                # trajs: batch_size, 2, seq_len; trajs_len: batch_size, 2
                trajs, trajs_len = [b.to(device) for b in batch]
                batch_size = trajs.shape[0]
                # trajs: batch_size * 2, seq_len; trajs_len: batch_size * 2
                trajs = trajs.view(-1, trajs.shape[-1])
                trajs_len = trajs_len.view(-1)

                features = self.model(trajs, trajs_len)
                logits, labels = self.info_nce_loss(features, batch_size)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            # save model checkpoints
            checkpoint_name = 'checkpoint_{:04d}_hiddensize_{}_batchsize_{}_bidirectional_{}_nlayers_{}_freeze_{}.pth.tar'.format(
                epoch_counter, self.args.hidden_size, self.args.batch_size, self.args.bidirectional, self.args.n_layers, self.args.freeze)
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.args.logdir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.args.logdir}.")

        logging.info("Training has finished.")

