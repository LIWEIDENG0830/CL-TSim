import shutil
import torch
import numpy as np
from prettytable import PrettyTable

def print_args(args):
    pt = PrettyTable()
    pt.field_names = ["Parameters", "Values"]
    for param, value in vars(args).items():
        pt.add_row([param, value])
    print(pt)


def load_pretrained(filepath, vocab_size):
    with open(filepath) as f:
        # jump the first line, which is the description of the embeddings.
        n_words, dim = [int(value) for value in f.readline().strip().split(" ")]
        embeddings = np.random.randn(vocab_size, dim)
        for line in f:
            word_vec = [float(value) for value in line.strip().split(" ")]
            embeddings[int(word_vec[0])] = word_vec[1:]
    return torch.tensor(embeddings, dtype=torch.float)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res