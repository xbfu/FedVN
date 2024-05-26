from copy import deepcopy

import torch
import torch.nn as nn


class Server(object):
    def __init__(self, train_clients, dataset, model, hidden, num_vn, device, logger):
        self.dataset = dataset
        self.train_clients = train_clients
        self.num_train_graphs = [len(client.trainset) for client in self.train_clients]
        self.coefficients = [num_train_graph / sum(self.num_train_graphs) for num_train_graph in self.num_train_graphs]
        self.metric = dataset.metric
        self.model = deepcopy(model)
        self.vn_embedding = nn.Parameter(torch.zeros(num_vn, hidden, device=device))
        self.score = torch.full([num_vn], 1 / num_vn).to(device)
        stat_dict = {'Accuracy': self.get_acc, 'MAE': self.get_mae}
        self.get_stat = stat_dict[self.metric]
        self.logger = logger

    def train(self, rounds):
        best_test_res = 10000 if self.metric == 'MAE' else 0.
        for r in range(1, rounds + 1):
            averaged_weights = {}
            global_vn_embedding = torch.zeros_like(self.vn_embedding)
            for i, client in enumerate(self.train_clients):
                client.set_parameters(self.model)
                client.set_vn_embedding(self.vn_embedding)
                weight, vn_embedding = client.local_update(self.score)
                global_vn_embedding += self.coefficients[i] * deepcopy(vn_embedding)
                for key in self.model.state_dict().keys():
                    if key in averaged_weights.keys():
                        averaged_weights[key] += self.coefficients[i] * deepcopy(weight)[key]
                    else:
                        averaged_weights[key] = self.coefficients[i] * deepcopy(weight)[key]

            self.model.load_state_dict(averaged_weights)
            self.vn_embedding.data = global_vn_embedding.data.clone()

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
            client.set_vn_embedding(self.vn_embedding)
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
            client.set_vn_embedding(self.vn_embedding)
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
