import torch
import random
import os
from torch import nn, autograd
import pickle as pkl
import numpy as np

class MLPBase(nn.Module):
    def __init__(self):
        super(MLPBase, self).__init__()
        lin1 = nn.Linear(2 * 14 * 14, 256)  # hidden_dimension = 256
        lin2 = nn.Linear(256, 256)
        lin3 = nn.Linear(256, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def evaluation(base_model):
    # Load Data
    f = open('env_storage.pkl', 'rb')
    envs = pkl.load(f)
    f.close()

    # Set model
    mlp = base_model.cuda()

    for env in envs:
        logits = mlp(env['images'])
        env['nll'] = mean_nll(logits, env['labels'])
        env['acc'] = mean_accuracy(logits, env['labels'])
        env['penalty'] = penalty(logits, env['labels'])

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

    test_nll = envs[2]['nll'].mean()
    test_acc = envs[2]['acc'].mean()
    test_penalty = envs[2]['penalty'].mean()

    weight_norm = torch.tensor(0.).cuda()
    for w in mlp.parameters():
        weight_norm += w.norm().pow(2)

    train_loss = train_nll.clone()
    train_loss += 0.001 * weight_norm  # l2_regularizer_weight * weight_norm
    penalty_weight = 10000.0  # penalty_weight after annel_iters
    train_loss += penalty_weight * train_penalty

    test_loss = test_nll.clone()
    test_loss += 0.001 * weight_norm  # l2_regularizer_weight * weight_norm
    penalty_weight = 10000.0  # penalty_weight after annel_iters #TODO: anneal하기 전후 구분 필요
    test_loss += penalty_weight * test_penalty

    if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        train_loss /= penalty_weight
        test_loss /= penalty_weight

    train = {'nll': train_nll, 'loss': train_loss, 'accuracy': train_acc}
    test = {'nll': test_nll, 'loss': test_loss, 'accuracy': test_acc}
    return train, test


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True