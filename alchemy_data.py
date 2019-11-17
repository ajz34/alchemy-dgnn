"""
Generate pytorch ready .pt files, and as database parser

Usage:
    alchemy_data.py [--task=<str>]

Options:
    --task=<str>  Task (dev, valid_00, test, ...) to generate .pt files [default: all]
"""

from docopt import docopt
import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def atom_dat_reader(atm_dict, target):
    # vertex
    atom_vertex = np.hstack([
        # atm_dict["atm_coord"],
        atm_dict["atm_charge"][:, None],
        atm_dict["atm_symbol_onehot"],
        atm_dict["atm_addcharge"][:, None],
        atm_dict["atm_aromatic"][:, None],
        atm_dict["atm_hybrid"],
    ])
    # edge index
    natm = atom_vertex.shape[0]
    natm_rangerep = np.arange(natm)[:, None].repeat(natm, axis=1)
    atom_edgeidx = np.array([natm_rangerep.flatten(), natm_rangerep.T.flatten()])
    # edge
    atom_edge = np.hstack([
        atm_dict["atm_dist"].reshape(-1)[:, None],
        atm_dict["atm_nuceng_adaj"].reshape(-1)[:, None],
        atm_dict["atm_edge_type"].reshape(5, -1).T,
    ])

    # construct atom-network data
    atom_data = Data(
        x=torch.as_tensor(atom_vertex, dtype=torch.float32),
        edge_index=torch.as_tensor(atom_edgeidx, dtype=torch.long),
        edge_attr=torch.as_tensor(atom_edge, dtype=torch.float32),
        y=torch.as_tensor(target, dtype=torch.float32)
    )
    return atom_data


def orbital_dat_reader(ao_dict, target):
    # vertex index (to its atom)
    orbital_vertindex = ao_dict["ao_idx"]
    # vertex
    orbital_vertex = np.hstack([
        # ao_dict["ao_coord"],
        ao_dict["ao_atomchg"][:, None],
        ao_dict["ao_atomhot"],
        ao_dict["ao_zeta"][:, None],
        ao_dict["ao_valence"][:, None],
        ao_dict["ao_spacial_x"][:, None],
        ao_dict["ao_spacial_y"][:, None],
        ao_dict["ao_spacial_z"][:, None],
    ])
    # edge index
    nao = orbital_vertex.shape[0]
    nao_rangerep = np.arange(nao)[:, None].repeat(nao, axis=1)
    orbital_edgeidx = np.array([nao_rangerep.flatten(), nao_rangerep.T.flatten()])
    # edge
    orbital_edge = np.hstack([
        ao_dict["ao_dist"].reshape(-1)[:, None],
        ao_dict["int1e_ovlp"].reshape(-1)[:, None],
        ao_dict["int1e_kin"].reshape(-1)[:, None],
        ao_dict["int1e_nuc"].reshape(-1)[:, None],
        ao_dict["int1e_r"].reshape(3, -1).T,
        ao_dict["rdm1e"].reshape(-1)[:, None],
    ])

    # construct atom-network data
    orbital_data = Data(
        x=torch.as_tensor(orbital_vertex, dtype=torch.float32),
        edge_index=torch.as_tensor(orbital_edgeidx, dtype=torch.long),
        edge_attr=torch.as_tensor(orbital_edge, dtype=torch.float32),
        y=torch.as_tensor(target, dtype=torch.float32),
        atom_idx=torch.as_tensor(orbital_vertindex, dtype=torch.long)
    )
    return orbital_data


class AlchemyData(InMemoryDataset):

    def __init__(self, mode="dev", net_type="atom", root_path=".", train_csv_path=None, transform=None, pre_transform=None):
        self.mode = mode
        self.net = net_type
        self.root_path = Path(root_path)
        self.train_csv_path = train_csv_path  # type: str or None
        self.target = NotImplemented

        super(AlchemyData, self).__init__(root_path, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.mode + ".dat"

    @property
    def processed_file_names(self):
        return self.net + "-" + self.mode + ".pt"

    def _download(self):
        pass

    def download(self):
        pass

    def process(self):

        process_count = 0

        if self.train_csv_path is not None:
            self.target = pd.read_csv(self.train_csv_path, index_col=0, usecols=['gdb_idx', ] + ['property_{}'.format(x) for x in range(12)])
            self.target = self.target[['property_{}'.format(x) for x in range(12)]]

        dat_file = self.raw_paths[0]
        with open(dat_file, "rb") as dat:
            dat_dict = pickle.load(dat)

        data_atom_list = []
        data_orbital_list = []

        for entry in dat_dict:
            # process_count += 1
            # if process_count % 10 == 0:
            #     print("processed " + str(process_count))
            target = torch.as_tensor(self.target.loc[entry].tolist(), dtype=torch.float32) \
                if self.target is not NotImplemented \
                else torch.as_tensor([entry], dtype=torch.float32)
            dat_atom, dat_orbital = dat_dict[entry]
            data_atom_list.append(atom_dat_reader(dat_atom, torch.as_tensor(target, dtype=torch.float32).unsqueeze(0)))
            data_orbital_list.append(orbital_dat_reader(dat_orbital, torch.as_tensor(target, dtype=torch.float32).unsqueeze(0)))

        data_atom, slices_atom = self.collate(data_atom_list)
        torch.save((data_atom, slices_atom), self.processed_dir + "/atom-" + self.mode + ".pt")
        data_orbital, slices_orbital = self.collate(data_orbital_list)
        torch.save((data_orbital, slices_orbital), self.processed_dir + "/orbital-" + self.mode + ".pt")


if __name__ == '__main__':

    arguments = docopt(__doc__)
    print(arguments)
    TASK = arguments["--task"]
    tasks = [TASK]
    if TASK == "all":
        tasks = ["dev", "test", "valid_00", "valid_01", "valid_02", "valid_03", "valid_04"]

    for t in tasks:
        print("processing task " + t + "...")
        if t == "test":
            train_csv_path = None
        else:
            train_csv_path = "./raw/train.csv"
        AlchemyData(mode=t, net_type="atom", train_csv_path=train_csv_path)
