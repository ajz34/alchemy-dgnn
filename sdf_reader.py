"""
Quantum chemistry calculation when reading original sdf database

Usage:
    sdf_reader.py [--task=<str>]

Options:
    --task=<str>  Task (dev, valid_00, train, ...) [default: all]
"""

from docopt import docopt
import numpy as np
from rdkit import Chem
from pyscf import gto, lib
import scipy


slater_zeta_valance = {
    "H": [[1.24], [1]],
    "C": [[5.67, 1.72, 1.72, 1.72, 1.72],
          [0, 1, 1, 1, 1]],
    "N": [[6.67, 1.95, 1.95, 1.95, 1.95],
          [0, 1, 1, 1, 1]],
    "O": [[7.66, 2.25, 2.25, 2.25, 2.25],
          [0, 1, 1, 1, 1]],
    "F": [[8.65, 2.55, 2.55, 2.55, 2.55],
          [0, 1, 1, 1, 1]],
    "S": [[15.47, 5.79, 5.79, 5.79, 5.79, 2.05, 2.05, 2.05, 2.05],
          [0, 0, 0, 0, 0, 1, 1, 1, 1]],
    "Cl": [[16.43, 6.26, 6.26, 6.26, 6.26, 2.10, 2.10, 2.10, 2.10],
           [0, 0, 0, 0, 0, 1, 1, 1, 1]],
}


def sdf_to_dict(sdf_path, mode="orbital"):
    with open(sdf_path, "r") as f:
        sdf_string = f.read()
    mol_sdf = Chem.MolFromMolBlock(sdf_string, removeHs=False)

    # -- Atomic Feature
    atm_dict = {}
    atm_coord = mol_sdf.GetConformers()[0].GetPositions() / lib.param.BOHR
    atm_dist = np.linalg.norm(atm_coord[None, :, :] - atm_coord[:, None, :], axis=-1)
    atm_charge = np.array([atm.GetAtomicNum() for atm in mol_sdf.GetAtoms()])
    atm_addcharge = np.array([atm.GetFormalCharge() for atm in mol_sdf.GetAtoms()])

    atm_nuceng_adaj = atm_dist + np.diag(np.ones(atm_dist.shape[0]) * np.inf)
    atm_nuceng_adaj = (atm_charge[None, :] * atm_charge[:, None]) / atm_nuceng_adaj

    atm_symbol = np.array([atm.GetSymbol() for atm in mol_sdf.GetAtoms()])
    atm_symbol_onehot = []
    for symbol in atm_symbol:
        atm_symbol_onehot.append([1 if symbol == x else 0 for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']])
    atm_symbol_onehot = np.array(atm_symbol_onehot)

    # Emperical descriptors
    atm_aromatic = np.array([int(atm.GetIsAromatic()) for atm in mol_sdf.GetAtoms()])
    atm_hybrid = np.array([[int(atm.GetHybridization() == x)
                            for x in (Chem.rdchem.HybridizationType.SP,
                                      Chem.rdchem.HybridizationType.SP2,
                                      Chem.rdchem.HybridizationType.SP3)]
                           for atm in mol_sdf.GetAtoms()])

    natm = mol_sdf.GetNumAtoms()
    atm_edge_type = np.zeros((5, natm, natm))
    for i in range(natm):
        for j in range(natm):
            e_ij = mol_sdf.GetBondBetweenAtoms(i, j)
            if e_ij is None:
                continue
            edge_type_vect = (e_ij.GetBondType() == np.array([Chem.rdchem.BondType.SINGLE,
                                                              Chem.rdchem.BondType.DOUBLE,
                                                              Chem.rdchem.BondType.TRIPLE,
                                                              Chem.rdchem.BondType.AROMATIC,
                                                              0]))
            if edge_type_vect.sum() == 0:
                edge_type_vect[4] = 1
            atm_edge_type[:, i, j] = edge_type_vect

    atm_dict["atm_coord"] = atm_coord  # Atomic position, unit Bohr, for backup (natm, 3)
    atm_dict["atm_dist"] = atm_dist  # Atomic distance matrix, unit in Bohr (natm, natm)
    atm_dict["atm_charge"] = atm_charge  # Atom charges, unit a.u. (natm, )
    atm_dict["atm_nuceng_adaj"] = atm_nuceng_adaj  # Nuculeu repulsion energy matrix, unit Eh (natm, natm)
    atm_dict["atm_symbol_onehot"] = atm_symbol_onehot  # Atom name in one-hot encoding (natm, 7)
    atm_dict["atm_addcharge"] = atm_addcharge  # WARNING: emperical value # Additional charge on atom (natm, )

    atm_dict["atm_aromatic"] = atm_aromatic  # WARNING: emperical value # Whether atom is aromatic (natm, )
    atm_dict["atm_hybrid"] = atm_hybrid  # WARNING: emperical value # Atom hybrid type (natm, 3)
    atm_dict["atm_edge_type"] = atm_edge_type  # WARNING: emperical value # Edge type (5, natm, natm)

    if mode == "atom":
        return atm_dict, None

    # -- Electronic Feature
    ao_dict = {}

    # construct molecule
    mol_list = []
    for atm, coord in zip(atm_symbol, atm_coord):
        atm_row = [atm] + [str(f) for f in coord.tolist()]
        mol_list.append(" ".join(atm_row))
    mol = gto.Mole()
    mol.atom = "\n".join(mol_list)
    mol.charge = atm_addcharge.sum()
    mol.basis = "STO-3G"
    mol.build()
    assert (mol.nelec[0] == mol.nelec[1])
    nocc = mol.nelec[0]

    # integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    P = mol.intor("int1e_r")

    # guess density
    e, C = scipy.linalg.eigh(T + V, S)
    D = C[:, :nocc] @ C[:, :nocc].T * 2

    # ao feature / definition
    ao_idx = np.zeros(mol.nao)
    ao_atomchg = np.zeros(mol.nao)
    ao_atomhot = np.zeros((mol.nao, 7))
    ao_zeta = np.zeros(mol.nao)
    ao_valence = np.zeros(mol.nao)
    ao_momentum = np.zeros(mol.nao)
    ao_spacial_x = np.zeros(mol.nao)
    ao_spacial_y = np.zeros(mol.nao)
    ao_spacial_z = np.zeros(mol.nao)
    ao_coord = np.zeros((mol.nao, 3))
    ao_dist = np.zeros((mol.nao, mol.nao))

    for atm, aoslice in zip(range(mol.natm), mol.aoslice_by_atom()):
        atm_symbol = mol.atom_symbol(atm)
        _, _, p0, p1 = aoslice
        ao_idx[p0:p1] = atm
        ao_atomchg[p0:p1] = mol.atom_charge(atm)
        ao_atomhot[p0:p1] = [1 if atm_symbol == x else 0 for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
        ao_coord[p0:p1] = mol.atom_coord(atm)
        ao_zeta[p0:p1], ao_valence[p0:p1] = slater_zeta_valance[atm_symbol]
        for atm2, aoslice2 in zip(range(mol.natm), mol.aoslice_by_atom()):
            _, _, p20, p21 = aoslice2
            ao_dist[p0:p1, p20:p21] = np.linalg.norm(mol.atom_coord(atm) - mol.atom_coord(atm2))

    ao_momentum[mol.search_ao_label("p")] = 1
    ao_spacial_x[mol.search_ao_label("x")] = 1
    ao_spacial_y[mol.search_ao_label("y")] = 1
    ao_spacial_z[mol.search_ao_label("z")] = 1

    # merge in ao_dict
    ao_dict["nelec"] = mol.nelec[0] * 2

    ao_dict["int1e_ovlp"] = S
    ao_dict["int1e_kin"] = T
    ao_dict["int1e_nuc"] = V
    ao_dict["int1e_r"] = P
    ao_dict["rdm1e"] = D

    ao_dict["ao_idx"] = ao_idx
    ao_dict["ao_atomchg"] = ao_atomchg
    ao_dict["ao_atomhot"] = ao_atomhot
    ao_dict["ao_zeta"] = ao_zeta
    ao_dict["ao_valence"] = ao_valence  # WARNING: emperical value
    ao_dict["ao_momentum"] = ao_momentum
    ao_dict["ao_spacial_x"] = ao_spacial_x
    ao_dict["ao_spacial_y"] = ao_spacial_y
    ao_dict["ao_spacial_z"] = ao_spacial_z
    ao_dict["ao_coord"] = ao_coord
    ao_dict["ao_dist"] = ao_dist

    return atm_dict, ao_dict


if __name__ == '__main__':

    from pathlib import Path
    import pickle

    arguments = docopt(__doc__)
    print(arguments)
    TASK = arguments["--task"]
    tasks = [TASK]
    if TASK == "all":
        tasks = ["dev", "test", "valid_00", "valid_01", "valid_02", "valid_03", "valid_04"]

    sdf, root = Path("raw-sdf"), Path("raw")
    for t in tasks:
        print(t)
        d = {}
        for f in (sdf / t).glob("**/*.sdf"):
            # print(f.name)
            r = sdf_to_dict(f)
            d[int(f.stem)] = r
        with open(root / (t + ".dat"), "wb") as dat_f:
            pickle.dump(d, dat_f, pickle.HIGHEST_PROTOCOL)
