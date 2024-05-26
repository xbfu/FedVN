import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from GOOD.networks.models.MolEncoders import AtomEncoder, BondEncoder


class MLPGIN(nn.Module):
    def __init__(self, emb_dim):
        super(MLPGIN, self).__init__()
        self.l1 = nn.Linear(emb_dim, 2 * emb_dim)
        self.bn = nn.BatchNorm1d(2 * emb_dim, track_running_stats=False)
        self.l2 = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.l2(x)

        return x


class MLPVirtual(nn.Module):
    def __init__(self, emb_dim, num_vn):
        super(MLPVirtual, self).__init__()
        self.l1 = nn.Linear(emb_dim, 2 * emb_dim)
        self.l2 = nn.Linear(2 * emb_dim, emb_dim)

        self.bn1 = nn.BatchNorm1d(2 * num_vn * emb_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_vn * emb_dim, track_running_stats=False)

    def forward(self, x):
        batch_size, num_vn, emb_dim = x.shape

        x = self.l1(x)
        x = self.bn1(x.reshape(batch_size, -1)).reshape(batch_size, num_vn, 2 * emb_dim)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x.reshape(batch_size, -1)).reshape(batch_size, num_vn, emb_dim)
        x = F.relu(x)

        return x


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = MLPGIN(emb_dim)
        # self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.eps = 0
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return x_j
        else:
            return F.relu(x_j + edge_attr)


class GINEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(GINEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, emb_dim)

        self.conv1 = GINConv(emb_dim)
        self.conv2 = GINConv(emb_dim)
        self.conv3 = GINConv(emb_dim)

        self.bn1 = nn.BatchNorm1d(emb_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(emb_dim, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(emb_dim, track_running_stats=False)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h0 = self.lin(x)

        h1 = self.conv1(h0, edge_index, edge_attr)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.5, training=self.training)

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.5, training=self.training)

        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.bn3(h3)
        h3 = F.dropout(h3, p=0.5, training=self.training)

        return h3


class GINMolEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(GINMolEncoder, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim)

        self.conv1 = GINConv(emb_dim)
        self.conv2 = GINConv(emb_dim)
        self.conv3 = GINConv(emb_dim)

        self.bn1 = nn.BatchNorm1d(emb_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(emb_dim, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(emb_dim, track_running_stats=False)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h0 = self.atom_encoder(x)

        h1 = self.conv1(h0, edge_index, edge_attr)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.5, training=self.training)

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.5, training=self.training)

        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.bn3(h3)
        h3 = F.dropout(h3, p=0.5, training=self.training)

        return h3


class Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(in_channels, hidden_channels)
        self.l2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, h):
        h = self.l1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.l2(h)
        return h


class scoringGIN(nn.Module):
    def __init__(self, dataset, hidden_channels, num_vn):
        super(scoringGIN, self).__init__()
        in_channels = dataset.num_node_features
        if dataset.dataset_type == 'mol':
            self.encoder = GINMolEncoder(in_channels, hidden_channels)
        else:
            self.encoder = GINEncoder(in_channels, hidden_channels)

        self.l1 = nn.Linear(hidden_channels, hidden_channels)
        self.l2 = nn.Linear(hidden_channels, num_vn)

    def forward(self, data):
        emb = self.encoder(data)

        h = self.l1(emb)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        score = self.l2(h)

        return score, emb


class vGINEncoder(nn.Module):
    def __init__(self, dataset_type, in_channels, emb_dim, num_vn):
        super(vGINEncoder, self).__init__()
        self.dataset_type = dataset_type
        if dataset_type == 'mol':
            self.atom_encoder = AtomEncoder(emb_dim)
        else:
            self.lin = nn.Linear(in_channels, emb_dim)

        self.conv1 = GINConv(emb_dim)
        self.conv2 = GINConv(emb_dim)
        self.conv3 = GINConv(emb_dim)

        self.mlp1 = MLPVirtual(emb_dim, num_vn)
        self.mlp2 = MLPVirtual(emb_dim, num_vn)

        self.bn1 = nn.BatchNorm1d(emb_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(emb_dim, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(emb_dim, track_running_stats=False)

    def forward(self, data, vn, score):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = batch[-1].item() + 1
        num_vn, emb_dim = vn.shape
        vn_embedding = vn.expand(batch_size, -1, -1)

        if self.dataset_type == 'mol':
            h0 = self.atom_encoder(x)
        else:
            h0 = self.lin(x)

        vn_neigh = score.unsqueeze(2) * h0.unsqueeze(1)
        vn_embedding_temp = global_add_pool(vn_neigh.transpose(0, 1), batch).transpose(0, 1) + vn_embedding
        h0 = h0 + (score[:, None] @ vn_embedding[batch]).squeeze()

        h1 = self.conv1(h0, edge_index, edge_attr)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.5, training=self.training)

        vn_embedding = F.dropout(self.mlp1(vn_embedding_temp), p=0.5, training=self.training)
        vn_neigh = score.unsqueeze(2) * h1.unsqueeze(1)
        vn_embedding_temp = global_add_pool(vn_neigh.transpose(0, 1), batch).transpose(0, 1) + vn_embedding
        h1 = h1 + (score[:, None] @ vn_embedding[batch]).squeeze()

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.5, training=self.training)

        vn_embedding = F.dropout(self.mlp2(vn_embedding_temp), p=0.5, training=self.training)
        h2 = h2 + (score[:, None] @ vn_embedding[batch]).squeeze()

        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.bn3(h3)
        h3 = F.dropout(h3, p=0.5, training=self.training)
        return h3


class vGIN(nn.Module):
    def __init__(self, dataset, hidden_channels, num_vn):
        super(vGIN, self).__init__()
        out_channels = dataset.num_classes if dataset.metric == 'Accuracy' else 1
        in_channels = dataset.num_node_features
        self.encoder = vGINEncoder(dataset.dataset_type, in_channels, hidden_channels, num_vn)
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels)

    def forward(self, data, vn_embedding, score=None):
        h = self.encoder(data, vn_embedding, score)
        h = global_mean_pool(h, data.batch)
        h = self.classifier(h)
        return h
