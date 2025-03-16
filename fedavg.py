import os
import argparse
from copy import deepcopy
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from sklearn import metrics

from model import vGIN, scoringGIN, GINEncoder, GINMolEncoder, Classifier

from data import load_data
from logger import Mylogger

criterion_dict = {'Accuracy': nn.CrossEntropyLoss(), 'MAE': nn.MSELoss()}


class GIN(nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GIN, self).__init__()
        out_channels = dataset.num_classes if dataset.metric == 'Accuracy' else 1
        in_channels = dataset.num_node_features
        if dataset.dataset_type == 'mol':
            self.encoder = GINMolEncoder(in_channels, hidden_channels)
        else:
            self.encoder = GINEncoder(in_channels, hidden_channels)
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels)

    def forward(self, data):
        h = self.encoder(data)
        h = global_mean_pool(h, data.batch)
        h = self.classifier(h)
        return h


class Server(object):
    def __init__(self, train_clients, dataset, model, hidden, num_vn, device, logger):
        self.dataset = dataset
        self.train_clients = train_clients
        self.num_train_graphs = [len(client.trainset) for client in self.train_clients]
        self.coefficients = [num_train_graph / sum(self.num_train_graphs) for num_train_graph in self.num_train_graphs]
        self.metric = dataset.metric
        self.model = deepcopy(model)
        stat_dict = {'Accuracy': self.get_acc, 'MAE': self.get_mae}
        self.get_stat = stat_dict[self.metric]
        self.logger = logger

    def train(self, rounds):
        best_test_res = 10000 if self.metric == 'MAE' else 0.
        for r in range(1, rounds + 1):
            averaged_weights = {}
            for i, client in enumerate(self.train_clients):
                client.set_parameters(self.model)
                weight = client.local_update()
                for key in self.model.state_dict().keys():
                    if key in averaged_weights.keys():
                        averaged_weights[key] += self.coefficients[i] * deepcopy(weight)[key]
                    else:
                        averaged_weights[key] = self.coefficients[i] * deepcopy(weight)[key]

            self.model.load_state_dict(averaged_weights)

            test_loss, test_res = self.get_stat(r)
            if self.metric == 'MAE':
                best_test_res = min(best_test_res, test_res)
            else:
                best_test_res = max(best_test_res, test_res)

        return best_test_res

    def get_acc(self, r):
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        for i, client in enumerate(self.train_clients):
            client.set_parameters(self.model)
            train_results, test_results = client.stats()

            loss, acc, length = train_results
            train_loss_list.append(loss)
            train_acc_list.append(acc)

            loss, acc, length = test_results
            test_loss_list.append(loss)
            test_acc_list.append(acc)

        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / len(train_acc_list)
        test_loss = sum(test_loss_list) / len(test_loss_list)
        test_acc = sum(test_acc_list) / len(test_acc_list)

        log_info = ''.join(['| Round:{:4d} '.format(r),
                            '| train_loss: {:7.5f} '.format(train_loss),
                            '| test_loss: {:7.5f} '.format(test_loss),
                            '| train_acc: {:7.5f} '.format(train_acc),
                            '| test_acc: {:7.5f} |'.format(test_acc)])
        self.logger.info(log_info)
        return test_loss, test_acc

    def get_mae(self, r):
        train_loss_list = []
        train_mae_list = []
        test_loss_list = []
        test_mae_list = []

        for i, client in enumerate(self.train_clients):
            client.set_parameters(self.model)
            train_results, test_results = client.stats()

            loss, mae = train_results
            train_loss_list.append(loss)
            train_mae_list.append(mae)

            loss, mae = test_results
            test_loss_list.append(loss)
            test_mae_list.append(mae)

        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_mae = sum(train_mae_list) / len(train_mae_list)
        test_loss = sum(test_loss_list) / len(test_loss_list)
        test_mae = sum(test_mae_list) / len(test_mae_list)

        log_info = ''.join(['| Round:{:4d} '.format(r),
                            '| train_loss: {:7.5f} '.format(train_loss),
                            '| test_loss: {:7.5f} '.format(test_loss),
                            '| train_mae: {:7.5f} '.format(train_mae),
                            '| test_mae: {:7.5f} |'.format(test_mae)])
        self.logger.info(log_info)
        return test_loss, test_mae


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
        self.model = GIN(dataset, args.hidden).to(device)
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

    def local_update(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for iteration, data in enumerate(self.train_loader):
                data = data.to(self.device)
                # score, _ = self.gin(data)
                # score = torch.sigmoid(score)
                out = self.model(data)
                if self.metric == 'Accuracy':
                    loss = self.criterion(out, data.y.squeeze().long())
                else:
                    loss = self.criterion(out, data.y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model.state_dict()

    def stats(self):
        train_results = self.get_stat(self.train_loader)
        test_results = self.get_stat(self.test_loader)

        return train_results, test_results

    def evaluate_acc(self, loader):
        self.model.eval()
        pred_list = []
        label_list = []
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, data.y.squeeze().long())
            total_loss += loss.item() * data.num_graphs
            pred_list.extend(out.argmax(1).tolist())
            label_list.extend(data.y.squeeze().tolist())
        avg_loss = total_loss / len(loader.dataset)
        acc = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
        return avg_loss, acc, len(loader.dataset)

    def evaluate_mae(self, loader):
        self.model.eval()
        errors = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            error = F.l1_loss(out, data.y)
            errors += error.item() * data.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        mae = errors / len(loader.dataset)

        return avg_loss, mae


def run(args, seed):
    arch_name = os.path.basename(__file__).split('.')[0]
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    filename = f'log/{args.dataset_name}_{arch_name}_{args.num_local_graphs}_{args.lr}_{args.epochs}_{args.domain}_{args.num_vn}_{args.lambda1}_{args.lambda2}_{args.t}_{seed}.log'
    logger = Mylogger(filename, formatter)

    print(' | Dataset:           ', args.dataset_name,
          '\n | Domain:            ', args.domain,
          '\n | num_local_graphs:  ', args.num_local_graphs,
          '\n | lr:                ', args.lr,
          '\n | epochs:            ', args.epochs,
          '\n | hidden:            ', args.hidden,
          '\n | batch_size:        ', args.batch_size,
          '\n | num_vn:            ', args.num_vn,
          '\n | lambda_1:          ', args.lambda1,
          '\n | lambda_2:          ', args.lambda2,
          '\n | gpu_id:            ', args.gpu_id,
          '\n | seed:              ', seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dataset, partitions = load_data(args.dataset_name, args.data_path, args.domain)
    model = GIN(dataset['train'], args.hidden).to(device)

    num_clients = len(partitions)
    client_ids = [i for i in range(num_clients)]
    client_list = [Client(client_id, partitions[client_id], args, device) for client_id in client_ids]

    server = Server(client_list, dataset['train'], model, args.hidden, args.num_vn, device, logger)
    best_test_res = server.train(rounds=args.rounds)
    log_info = ''.join(['| Arch: {:s} '.format(arch_name),
                        '| dataset: {:s} '.format(args.dataset_name),
                        '| lr: {:6.4f} '.format(args.lr),
                        '| epochs: {:2d} '.format(args.epochs),
                        '| domain: {:s} '.format(args.domain),
                        '| seed: {:2d} '.format(seed),
                        '| best_test_res: {:7.5f} |'.format(best_test_res)])
    logger.info(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The code of FedVN')
    parser.add_argument('--dataset_name', type=str, default='motif', help='dataset used for training')
    parser.add_argument('--domain', type=str, default='basis', help='data partition setting')
    parser.add_argument('--data_path', type=str, default='./data/', help='data directory')
    parser.add_argument('--num_local_graphs', type=int, default=1000, help='number of local graphs (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--rounds', type=int, default=100, help='number of rounds (default: 100)')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in the model (default: 100)')
    parser.add_argument('--hidden_eg', type=int, default=100, help='hidden size in the edge generator (default: 100)')
    parser.add_argument('--num_vn', type=int, default=20, help='number of virtual nodes (default: 20)')
    parser.add_argument('--lambda1', type=float, default=0.1, help='the value of lambda1 (default: 0.1)')
    parser.add_argument('--lambda2', type=float, default=1.0, help='the value of lambda2 (default: 1.0)')
    parser.add_argument('--t', type=float, default=0.1, help='temperature (default: 0.1)')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU device ID (default: 0)')
    args = parser.parse_args()

    num_vn = args.num_vn
    for seed in range(5):
        run(args, seed)
