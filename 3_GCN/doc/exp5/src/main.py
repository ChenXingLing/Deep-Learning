
## ——————【config】—————— ##

default_cfg = {
    'data_root': './../GNN/',
    'data_name': 'cora',
    'num_train_per_class': 20,
    'num_val': 500,
    'num_test': 1000,
    'seed': 114514,
    'device': 'cuda:0',
    'epochs': 1000,
    'patience': 5,
    'lr': 5e-3,
    'weight_decay': 5e-4,
    'hidden_dim': 32,
    'n_layers': 2,
    'activations': 'relu',
    'dropout': 0.5,
    'drop_edge': 0.,
    'add_self_loop': True,
    'pair_norm': False,
    'test_only': False
}


class Config(object):
    def __init__(self, ):
        self.data_root = None
        self.data_name = None
        self.num_train_per_class = None
        self.num_val = None
        self.num_test = None
        self.seed = None
        self.device = None
        self.epochs = None
        self.patience = None
        self.lr = None
        self.weight_decay = None
        self.hidden_dim = None
        self.n_layers = None
        self.activations = None
        self.dropout = None
        self.drop_edge = None
        self.add_self_loop = None
        self.pair_norm = None
        self.test_only = None
        self.reset()

    def reset(self):
        for key, val in default_cfg.items():
            setattr(self, key, val)

    def update(self, new_cfg):
        for key, val in new_cfg.items():
            setattr(self, key, val)


## ——————【utils】—————— ##

import torch
import random
import os
import numpy as np


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"



## ——————【data】—————— ##

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected

classes = {
    'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
    'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods',
             'Reinforcement_Learning', 'Rule_Learning', 'Theory']
}


class NodeClsDataset(InMemoryDataset):
    def __init__(self, root, name, num_train_per_class: int = 20,
                 num_val: int = 500, num_test: int = 1000, transform=None):
        self.name = name.lower()
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        super(NodeClsDataset, self).__init__(root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}.content', f'{self.name}.cites']

    @property
    def processed_file_names(self):
        return [f'{self.name}.pt']

    def download(self):
        pass

    def process(self):
        label2index = {label: i for i, label in enumerate(classes[f'{self.name}'])}
        id2index, x, y = read_content(self.raw_paths[0], label2index)
        edge_index = read_cites(self.raw_paths[1], id2index)
        data = Data(x=x, y=y, edge_index=edge_index)

        data.train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        for c in range(len(label2index)):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:self.num_train_per_class]]
            data.train_mask[idx] = True

        remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        data.val_mask[remaining[:self.num_val]] = True
        data.test_mask[remaining[self.num_val:self.num_val + self.num_test]] = True

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


def read_content(content_file, label2index):
    with open(content_file, 'r') as f:
        lines = f.read().strip().split('\n')[:-1]
    id2index = {}
    x = []
    y = []
    for i, line in enumerate(lines):
        line = line.strip().split('\t')
        paper_id, attr, label = line[0], line[1:-1], line[-1]
        id2index[paper_id] = i
        x.append([float(e) for e in attr])
        y.append(label2index[label])
    return id2index, torch.tensor(x), torch.tensor(y, dtype=torch.long)


def read_cites(cites_file, id2index):
    with open(cites_file, 'r') as f:
        lines = f.read().strip().split('\n')[:-1]
    edge_index = []
    for line in lines:
        cited, citing = line.strip().split('\t')
        if citing not in id2index or cited not in id2index:
            continue
        id_cited, id_citing = id2index[cited], id2index[citing]
        edge_index.append([id_citing, id_cited])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = to_undirected(edge_index)
    return edge_index.t().contiguous()


## ——————【Model】—————— ##

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils import dropout_adj


activations = {
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
}


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 n_layers: int, act: str = 'relu', add_self_loops: bool = True,
                 pair_norm: bool = True, dropout: float = .0, drop_edge: float = .0):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.drop_edge = drop_edge
        self.pair_norm = pair_norm
        self.act = activations[act] if isinstance(act, str) else act

        self.conv_list = torch.nn.ModuleList()
        for i in range(n_layers):
            in_c, out_c = hidden_channels, hidden_channels
            if i == 0:
                in_c = in_channels
            elif i == n_layers - 1:
                out_c = num_classes
            self.conv_list.append(GCNConv(in_c, out_c, add_self_loops=add_self_loops))

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=self.drop_edge)

        for i, conv in enumerate(self.conv_list):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = PairNorm()(x)
            if i < len(self.conv_list) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

## ——————【main】—————— ##

import torch.nn as nn
from torch.optim import Adam
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from itertools import product
import pandas as pd


def train(model, data, optimizer, loss_fc):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = loss_fc(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Get the predictions
    preds = torch.argmax(logits, dim=1).flatten()
    acc = (preds[data.train_mask] == data.y[data.train_mask]).cpu().numpy().mean()

    return loss, acc


def evaluate(model, data, loss_fc, mode='val'):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask = getattr(data, f'{mode}_mask')
        loss = loss_fc(logits[mask], data.y[mask])
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        acc = (preds[mask] == data.y[mask]).cpu().numpy().mean()

    return loss, acc


def main(cfg: Config):
    set_seed(cfg.seed)
    dataset = NodeClsDataset(cfg.data_root, cfg.data_name, cfg.num_train_per_class,
                             cfg.num_val, cfg.num_test, transform=NormalizeFeatures())
    # from torch_geometric.datasets import Planetoid
    # dataset = Planetoid(root='./tmp/Cora', name='Cora', split='random', transform=NormalizeFeatures())

    model = GCN(dataset.num_node_features, cfg.hidden_dim, dataset.num_classes,
                n_layers=cfg.n_layers, act=cfg.activations, add_self_loops=cfg.add_self_loop,
                pair_norm=cfg.pair_norm, dropout=cfg.dropout, drop_edge=cfg.drop_edge)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    data = dataset[0].to(cfg.device)
    model = model.to(device=cfg.device)
    criterion = criterion.to(cfg.device)
    if not cfg.test_only:
        best_valid_loss = np.inf
        wait = 0
        for epoch in range(cfg.epochs):
            print(f">>> Epoch {epoch + 1}/{cfg.epochs}")

            train_loss, train_acc = train(model, data, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, data, criterion, mode='val')

            print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc * 100:.2f}%')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                wait = 0
                torch.save(model.state_dict(), './checkpoint/best_weights.pt')
            else:
                wait += 1
                if wait > cfg.patience:
                    print('>>> Early stopped.')
                    break

    print(">>> Testing...")
    model.load_state_dict(torch.load("./checkpoint/best_weights.pt"))
    test_loss, test_acc = evaluate(model, data, criterion, mode='test')
    print(f'\tTest Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')
    return test_acc


if __name__ == '__main__':
    config = Config()
    # main(config)
    # exit()

    cfg_grid = {
        'data_name': ['citeseer', 'cora'],
        'add_self_loop': [True, False],
        'n_layers': [1, 2, 3, 5, 10],
        'drop_edge': [0, .1, .2, .3, .5],
        'pair_norm': [True, False],
        'activations': ['relu', 'tanh', 'sigmoid']
    }
    results = []
    keys = cfg_grid.keys()
    for values in product(*cfg_grid.values()):
        new_cfg = dict(zip(keys, values))
        config.update(new_cfg)
        acc = main(config)
        results.append([*new_cfg.values, acc])
    df = pd.DataFrame(results, columns=[*cfg_grid.keys(), 'test_acc'])
    df.to_csv('./result.csv', index=False)
