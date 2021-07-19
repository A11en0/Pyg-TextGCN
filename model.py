import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv  # noqa
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached=True, normalize=True)
        self.conv2 = GCNConv(nhid, nclass, cached=True, normalize=True)
        # self.conv1 = ChebConv(nfeat, nhid, K=2)
        # self.conv2 = ChebConv(nhid, nclass, K=2)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, heads=8, dropout=dropout)
        self.conv2 = GATConv(nhid * 8, nclass, heads=1, concat=False,
                             dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)