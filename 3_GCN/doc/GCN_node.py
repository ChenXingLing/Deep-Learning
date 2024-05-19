import sys
import random
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, TransformerConv
import torch.optim as optim


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
setup_seed(42)

# 加载数据集
def load_dataset(name):
    dataset = Planetoid(root='dataset/' + name, name=name, transform=T.NormalizeFeatures())
    return dataset

# 使用Tsne绘图
def plot_points(z, y):
    z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    classes = len(torch.unique(y))
    y = y.cpu().numpy()
    plt.figure(figsize=(8, 8))
    for i in range(classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=args.tsne_colors[i])
    plt.axis('off')
    plt.savefig('{} embeddings ues tnse to plt figure.png'.format(args.model))
    plt.show()

# 定义模型
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits

if __name__ == "__main__":

    dataset='Cora' #"Dataset selection"
    hidden_dim=32 #'Hidden layer dimension'
    dropout_rate=0.5 #'Dropout rate'
    lr=0.01 #"Learning Rate selection"
    wd=5e-4 #"weight_decay selection"
    epochs=200 #="train epochs selection"
    
    tsne_drawing=False #"Whether to use tsne drawing"
    tsne_colors=['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'] #"colors"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset)
    data = dataset[0].to(device)
    print(data)
    # sys.exit()
    model = GNN(in_channels=dataset.num_node_features, hidden_channels=hidden_dim,
                out_channels=dataset.num_classes,  dropout_rate=dropout_rate).to(device)
    print(model)
    print(f"Loaded {dataset} dataset with {data.num_nodes} nodes and {data.num_edges} edges.")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    Best_Acc = []
    for epoch in range(1, epochs):
        loss = train(model, data)
        accs, log= test(model, data)
        train_acc, val_acc, test_acc = accs
        print(f'Epoch: [{epoch:03d}/200], Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        Best_Acc.append(test_acc)
    if tsne_drawing == True:
        plot_points(log, data.y)
    print('---------------------------')
    print('Best Acc: {:.4f}'.format(test_acc))
    print('---------------------------')