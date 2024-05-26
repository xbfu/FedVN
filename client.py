import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from model import vGIN, scoringGIN

criterion_dict = {'Accuracy': nn.CrossEntropyLoss(), 'MAE': nn.MSELoss()}


class Client(object):
    def __init__(self, client_id, dataset, args, device):
        self.client_id = client_id
        self.dataset = dataset
        self.avg_size = dataset.x.shape[0] / len(dataset)
        self.trainset, self.testset = self.split_train_test(args.num_local_graphs)
        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.lr = args.lr
        self.epochs = args.epochs
        self.device = device
        self.metric = dataset.metric
        self.num_classes = dataset.num_classes
        self.model = vGIN(dataset, args.hidden, args.num_vn).to(device)
        self.num_vn = args.num_vn
        self.vn_embedding = nn.Parameter(torch.zeros(args.num_vn, args.hidden, device=device), requires_grad=True)
        self.optimizer_vn = torch.optim.SGD([self.vn_embedding], lr=args.lr)
        self.gin = scoringGIN(dataset, args.hidden_eg, args.num_vn).to(device)
        self.optimizer_per = torch.optim.SGD(self.gin.parameters(), lr=args.lr)
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.t = args.t
        self.local_score = torch.full([args.num_vn], 0.5).to(self.device)
        self.criterion = criterion_dict[self.metric]
        stat_dict = {'Accuracy': self.evaluate_acc, 'MAE': self.evaluate_mae}
        self.get_stat = stat_dict[self.metric]

    def split_train_test(self, num_local_graphs):
        graph_ids = list(range(len(self.dataset)))
        np.random.shuffle(graph_ids)
        trainset = self.dataset[graph_ids[: int(0.8 * num_local_graphs)]]
        testset = self.dataset[graph_ids[int(0.8 * num_local_graphs):num_local_graphs]]
        return trainset, testset

    def set_parameters(self, model):
        for (_, new_param), (name, old_param) in zip(model.named_parameters(), self.model.named_parameters()):
            old_param.data = new_param.data.clone()

    def set_vn_embedding(self, vn_embedding):
        self.vn_embedding.data = vn_embedding.data.clone()

    def local_update(self, global_score):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.gin.train()
        self.model.eval()
        for epoch in range(1, self.epochs + 1):
            for iteration, data in enumerate(self.train_loader):
                data = data.to(self.device)
                score, _ = self.gin(data)
                score = torch.sigmoid(score)
                out = self.model(data, self.vn_embedding, score)
                if self.metric == 'Accuracy':
                    loss = self.criterion(out, data.y.squeeze().long())
                else:
                    loss = self.criterion(out, data.y)
                graph_score = global_add_pool(score, data.batch)
                sim_local = F.cosine_similarity(graph_score, self.local_score)
                sim_global = F.cosine_similarity(graph_score, global_score)
                t = self.t
                l_c = - torch.log(torch.exp(sim_local/t) / (torch.exp(sim_local/t) + torch.exp(sim_global/t))).mean()

                loss += self.lambda2 * l_c
                self.optimizer_per.zero_grad()
                optimizer.zero_grad()
                self.optimizer_vn.zero_grad()
                loss.backward()
                self.optimizer_per.step()

                self.local_score = graph_score.mean(0).detach()

        self.model.train()
        self.gin.eval()
        for epoch in range(1, self.epochs + 1):
            for iteration, data in enumerate(self.train_loader):
                data = data.to(self.device)
                score, _ = self.gin(data)
                score = torch.sigmoid(score)
                out = self.model(data, self.vn_embedding, score)
                if self.metric == 'Accuracy':
                    loss = self.criterion(out, data.y.squeeze().long())
                else:
                    loss = self.criterion(out, data.y)

                normalized_vn = F.normalize(self.vn_embedding - self.vn_embedding.mean(1)[:, None])
                correlation = normalized_vn @ normalized_vn.t()
                loss_f = correlation.pow(2).mean()

                loss += self.lambda1 * loss_f

                self.optimizer_per.zero_grad()
                optimizer.zero_grad()
                self.optimizer_vn.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_vn.step()

        return self.model.state_dict(), self.vn_embedding

    def stats(self):
        train_results = self.get_stat(self.train_loader)
        test_results = self.get_stat(self.test_loader)

        return train_results, test_results

    def evaluate_acc(self, loader):
        self.gin.eval()
        self.model.eval()
        pred_list = []
        label_list = []
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            score, _ = self.gin(data)
            score = torch.sigmoid(score)
            out = self.model(data, self.vn_embedding, score)
            loss = self.criterion(out, data.y.squeeze().long())
            total_loss += loss.item() * data.num_graphs
            pred_list.extend(out.argmax(1).tolist())
            label_list.extend(data.y.squeeze().tolist())
        avg_loss = total_loss / len(loader.dataset)
        acc = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
        return avg_loss, acc, len(loader.dataset)

    def evaluate_mae(self, loader):
        self.gin.eval()
        self.model.eval()
        errors = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            score, _ = self.gin(data)
            score = torch.sigmoid(score)
            out = self.model(data, self.vn_embedding, score)
            loss = self.criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            error = F.l1_loss(out, data.y)
            errors += error.item() * data.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        mae = errors / len(loader.dataset)

        return avg_loss, mae
