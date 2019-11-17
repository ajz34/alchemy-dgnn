"""
MPNN model with ensemble and cross validation

This program only do one validation, i.e., do not make validation ensemble.
Ensemble in program name means model ensemble, i.e., ensemble of atomic and orbital feature model.

Usage:
    train_multinet.py [--valid=<int>] [--early-stop=<int>] [--max-epoch=<int>] [--batch-size=<int>]

Options:
    --valid=<int>       Cross validation set index [default: 0]
    --early-stop=<int>  Early stop epoch threshold [default: 25]
    --max-epoch=<int>   Maximum epoches [default: 1000]
    --batch-size=<int>  Batch size in gradient descent [default: 16]
"""

from docopt import docopt
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
import time

from torch_geometric.data import DataLoader
import torch_geometric.transforms as transform
from torch.utils.data import RandomSampler

from alchemy_data import AlchemyData
from mpnn_multinet import MultiNet


class MyRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        self.random_seed = None
        super(MyRandomSampler, self).__init__(data_source, replacement, num_samples)

    def __iter__(self):
        torch.manual_seed(self.random_seed)
        return super(MyRandomSampler, self).__iter__()


def train(model, atom_loader, orbital_loader):
    model.train()

    loss_all = 0
    batch_count = 0
    data_number_count = 0
    timeorig = time.time()
    time0 = time.time()
    loaders_count = len(atom_loader)
    orbital_loader.sampler.random_seed = atom_loader.sampler.random_seed = np.random.randint(1, np.long(1e10))
    for data_atom_list, data_orbital_list in zip(atom_loader, orbital_loader):
        # Batch information dump
        time1 = time.time()
        if batch_count > 0:
            log_line = "Train batch: {0:}/{1:}, progress: {2:5.2f}%, batch time: {3:5.2f} s, estimate: {4:7.2f} s, current loss: {5:7.6f}".format(
                batch_count, loaders_count,
                batch_count / loaders_count * 100,
                time1 - time0,
                (time1 - timeorig) / batch_count * (loaders_count - batch_count),
                loss_all / data_number_count)
            print(log_line, end="\r")
        time0 = time1
        data_number_count += data_atom_list.num_graphs
        batch_count += 1
        # Optimize
        optimizer.zero_grad()
        data_atom_list, data_orbital_list = data_atom_list.to(device), data_orbital_list.to(device)
        y_model = model(data_atom_list, data_orbital_list)
        loss = nn.L1Loss()(y_model, data_atom_list.y)
        loss.backward()
        loss_all += loss.item() * data_atom_list.num_graphs
        optimizer.step()
        torch.cuda.empty_cache()
    print("", end="\r")
    print()
    return loss_all / len(atom_loader.dataset)


def valid(model, atom_loader, orbital_loader):
    model.eval()

    loss_all = 0
    with torch.no_grad():
        for data_atom_list, data_orbital_list in zip(atom_loader, orbital_loader):
            data_atom_list, data_orbital_list = data_atom_list.to(device), data_orbital_list.to(device)
            y_pred = model(data_atom_list, data_orbital_list)
            loss = nn.L1Loss()(y_pred, data_atom_list.y)
            loss_all += loss.item() * data_atom_list.num_graphs
        mean_loss = loss_all / len(atom_loader.dataset)
        return mean_loss


def test(model, atom_loader, orbital_loader):
    model.eval()

    with torch.no_grad():
        targets = dict()
        for data_atom_list, data_orbital_list in zip(atom_loader, orbital_loader):
            data_atom_list, data_orbital_list = data_atom_list.to(device), data_orbital_list.to(device)
            y_pred = model(data_atom_list, data_orbital_list)
            for i in range(y_pred.size()[0]):
                targets[data_atom_list.y[i].item()] = y_pred[i].tolist()
        return targets


def dump_test(ensemble, atom_loader, orbital_loader):
    targets = test(ensemble, atom_loader, orbital_loader)
    df_targets = pd.DataFrame.from_dict(targets, orient="index", columns=['property_%d' % x for x in range(TARGET_DIM)])
    df_targets.sort_index(inplace=True)
    df_targets.to_csv('targets_valid_{:02d}.csv'.format(CURRENT_VALID_ID), index_label='gdb_idx')


class AtomTransform(object):
    def __call__(self, data):
        edge_attr = data.edge_attr
        atom_edge = torch.zeros((edge_attr.shape[0], 10))
        atom_edge[:, :7] = edge_attr
        r = edge_attr[:, 0]
        rmask = (torch.abs(r) > 1e-7)
        atom_edge[:, 7][rmask], atom_edge[:, 8][rmask], atom_edge[:, 9][rmask] = 1 / r[rmask] * 5, torch.exp(- r[rmask]) * 50, 1 / r[rmask]**6 * 10
        atom_edge[:, 0] /= 25
        data.edge_attr = atom_edge
        return data


class OrbitalTransform(object):
    def __call__(self, data):
        x = data.x
        orbital_vertex = torch.zeros((x.shape[0], 21))
        orbital_vertex[:, :13] = x
        zeta = x[:, 8]
        for idx, multiplier, scaler in zip(range(8),
                                           [1, 1.5, 2, 2.5, 3, 4, 6, 9],
                                           [10, 15, 25, 50, 100, 250, 2500, 100000]):
            orbital_vertex[:, idx + 13] = torch.exp(- zeta * multiplier) * scaler
        data.x = orbital_vertex

        edge_attr = data.edge_attr
        orbital_edge = torch.zeros((edge_attr.shape[0], 10))
        orbital_edge[:, 1] = edge_attr[:, 1]  # int1e_ovlp
        orbital_edge[:, 2] = edge_attr[:, 2] / 25  # int1e_kin
        orbital_edge[:, 3] = edge_attr[:, 3]  # int1e_nuc
        orbital_edge[:, 4:7] = edge_attr[:, 4:7] / 25  # int1e_r
        orbital_edge[:, 7] = edge_attr[:, 7]  # rdm1e

        orbital_edge[:, 0] = edge_attr[:, 0]
        r = orbital_edge[:, 0]
        rmask = (torch.abs(r) > 1e-7)
        orbital_edge[:, 8][rmask], orbital_edge[:, 9][rmask] = 1 / r[rmask] * 5, torch.exp(- r[rmask]) * 50
        orbital_edge[:, 0] /= 25
        data.edge_attr = orbital_edge
        return data


if __name__ == '__main__':

    # Define important variables
    # CURRENT_VALID_ID = 0
    # EARLY_STOP_EPOCH_NUM = 25
    # MAX_EPOCH = 1000
    # BATCH_SIZE = 8
    VALID_DATASET_NUMBERS = 5
    TARGET_DIM = 12

    arguments = docopt(__doc__)
    print(arguments)
    CURRENT_VALID_ID = int(arguments["--valid"])
    EARLY_STOP_EPOCH_NUM = int(arguments["--early-stop"])
    MAX_EPOCH = int(arguments["--max-epoch"])
    BATCH_SIZE = int(arguments["--batch-size"])
    # Logging
    log = open("valid_{:02d}.log".format(CURRENT_VALID_ID), "w")

    # Prepare device
    if torch.cuda.is_available():
        print("torch.cuda.get_device_capability", torch.cuda.get_device_capability())
        print("torch.cuda.device_count", torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset definition
    valid_atom_dataset_list = []
    valid_orbital_dataset_list = []
    atom_transform = transform.Compose([AtomTransform()])
    orbital_transfrom = transform.Compose([OrbitalTransform()])

    for valid_idx in range(VALID_DATASET_NUMBERS):
        valid_atom_dataset_list.append(AlchemyData(mode='valid_{:02d}'.format(valid_idx), net_type="atom", train_csv_path="./raw/train.csv", transform=atom_transform))
        valid_orbital_dataset_list.append(AlchemyData(mode='valid_{:02d}'.format(valid_idx), net_type="orbital", train_csv_path="./raw/train.csv", transform=orbital_transfrom))
    dev_atom_dataset = AlchemyData(mode='dev', net_type="atom", train_csv_path="./raw/train.csv", transform=atom_transform)
    dev_orbital_dataset = AlchemyData(mode='dev', net_type="orbital", train_csv_path="./raw/train.csv", transform=orbital_transfrom)
    test_atom_dataset = AlchemyData(mode='test', net_type="atom", transform=atom_transform)
    test_orbital_dataset = AlchemyData(mode='test', net_type="orbital", transform=orbital_transfrom)

    test_atom_loader = DataLoader(test_atom_dataset, batch_size=BATCH_SIZE)
    test_orbital_loader = DataLoader(test_orbital_dataset, batch_size=BATCH_SIZE)

    # Prepare cross validation sets, which is actually used in dev/valid/test process
    train_atom_datasets, train_orbital_datasets = [dev_atom_dataset], [dev_orbital_dataset]
    valid_atom_dataset, valid_orbital_dataset = NotImplemented, NotImplemented
    for i in range(VALID_DATASET_NUMBERS):
        if i == CURRENT_VALID_ID:
            valid_atom_dataset, valid_orbital_dataset = valid_atom_dataset_list[i], valid_orbital_dataset_list[i]
        else:
            train_atom_datasets.append(valid_atom_dataset_list[i])
            train_orbital_datasets.append(valid_orbital_dataset_list[i])
    train_atom_dataset = torch.utils.data.ConcatDataset(train_atom_datasets)
    train_orbital_dataset = torch.utils.data.ConcatDataset(train_orbital_datasets)

    train_atom_sampler = MyRandomSampler(train_atom_dataset)
    train_orbital_sampler = MyRandomSampler(train_orbital_dataset)

    train_atom_loader = DataLoader(train_atom_dataset, batch_size=BATCH_SIZE, sampler=train_atom_sampler)
    train_orbital_loader = DataLoader(train_orbital_dataset, batch_size=BATCH_SIZE, sampler=train_orbital_sampler)
    valid_atom_loader = DataLoader(valid_atom_dataset, batch_size=BATCH_SIZE)
    valid_orbital_loader = DataLoader(valid_orbital_dataset, batch_size=BATCH_SIZE)

    # Define models
    model = MultiNet(
        atom_vertex_dim=(dev_atom_dataset.num_node_features, 18),
        atom_edge_dim=(dev_atom_dataset.num_edge_features, 12),
        orbital_vertex_dim=(dev_orbital_dataset.num_node_features, 12),
        orbital_edge_dim=(dev_orbital_dataset.num_edge_features, 8),
        output_dim=TARGET_DIM,
        mp_step=6,
        s2s_step=6
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    parameter_estimate = 0
    print(model)
    for param_name, param_tensor in model.state_dict().items():
        print("{:50} {:15} {:}".format(param_name, str(tuple(param_tensor.shape)), np.array(param_tensor.shape).prod()))
        log.write("{:50} {:15} {:}".format(param_name, str(tuple(param_tensor.shape)), np.array(param_tensor.shape).prod()) + "\n")
        parameter_estimate += np.array(param_tensor.shape).prod()
    print("Estimated parameter numbers: " + str(parameter_estimate))
    log.write("Estimated parameter numbers: " + str(parameter_estimate) + "\n")
    log.flush()

    # Loop epoch
    lowest_valid_loss = 1.e+10
    lowest_valid_epoch = 0

    for epoch in range(MAX_EPOCH):
        time0 = time.time()
        loss = train(model, train_atom_loader, train_orbital_loader)
        valid_loss = valid(model, valid_atom_loader, valid_orbital_loader)
        print('Epoch: {:03d}, Loss: {:.7f}, time: {:7.2f} s'.format(epoch, loss, time.time() - time0))
        print("Current validation set L1 error: {:10.6f}".format(valid_loss))
        log.write('Epoch: {:03d}, Loss: {:.7f}, time: {:7.2f} s'.format(epoch, loss, time.time() - time0) + "\n")
        log.write("            Current validation set L1 error: {:10.6f}".format(valid_loss) + "\n")
        if valid_loss < lowest_valid_loss:
            lowest_valid_loss = valid_loss
            lowest_valid_epoch = epoch
            print("Lowest validation set loss hit, testing begins...")
            log.write("Lowest validation set loss hit, testing begins...\n")
            dump_test(model, test_atom_loader, test_orbital_loader)
            torch.save(model.state_dict(), "model_valid_{:02d}.pt".format(CURRENT_VALID_ID))
        if epoch - lowest_valid_epoch > EARLY_STOP_EPOCH_NUM:
            print("Early stopping condition hit after " + str(EARLY_STOP_EPOCH_NUM) + " epochs without validation loss descend.")
            log.write("Early stopping condition hit after " + str(EARLY_STOP_EPOCH_NUM) + " epochs without validation loss descend.\n")
            break
        log.flush()

