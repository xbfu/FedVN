import os
import argparse
import logging
from logger import Mylogger
import numpy as np
import torch

from data import load_data
from model import vGIN
from server import Server
from client import Client


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
    model = vGIN(dataset['train'], args.hidden, args.num_vn).to(device)

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
    for seed in [1]:
        run(args, seed)
