import torch
from torch import nn
from torch_geometric.nn import NNConv, Set2Set


class MultiNet(torch.nn.Module):
    def __init__(self,
                 atom_vertex_dim,
                 atom_edge_dim,
                 orbital_vertex_dim=NotImplemented,
                 orbital_edge_dim=NotImplemented,
                 output_dim=NotImplemented,
                 mp_step=6,
                 s2s_step=6):
        super(MultiNet, self).__init__()
        self.atom_vertex_dim = atom_vertex_dim
        self.atom_edge_dim = atom_edge_dim
        self.orbital_vertex_dim = orbital_vertex_dim
        self.orbital_edge_dim = orbital_edge_dim
        self.output_dim = output_dim
        self.mp_step = mp_step
        self.s2s_step = s2s_step

        # atom net
        atom_edge_gc = nn.Sequential(nn.Linear(atom_edge_dim[1], atom_vertex_dim[1] ** 2), nn.Dropout(0.2))

        self.atom_vertex_conv = NNConv(atom_vertex_dim[1], atom_vertex_dim[1], atom_edge_gc, aggr="mean", root_weight=True)
        self.atom_vertex_gru = nn.GRU(atom_vertex_dim[1], atom_vertex_dim[1])

        self.atom_s2s = Set2Set(atom_vertex_dim[1], processing_steps=s2s_step)
        self.atom_lin0 = nn.Sequential(nn.Linear(atom_vertex_dim[0], 2 * atom_vertex_dim[0]), nn.CELU(),
                                       nn.Linear(2 * atom_vertex_dim[0], atom_vertex_dim[1]), nn.CELU())
        self.atom_lin1 = nn.Sequential(nn.Linear(atom_edge_dim[0], 2 * atom_edge_dim[0]), nn.CELU(),
                                       nn.Linear(2 * atom_edge_dim[0], atom_edge_dim[1]), nn.CELU())
        self.atom_lin2 = nn.Sequential(nn.Linear(2 * atom_vertex_dim[1], 4 * atom_vertex_dim[1]), nn.CELU())

        # orbital net
        orbital_edge_gc = nn.Sequential(nn.Linear(orbital_edge_dim[1], orbital_vertex_dim[1] ** 2), nn.Dropout(0.2))

        self.orbital_vertex_conv = NNConv(orbital_vertex_dim[1], orbital_vertex_dim[1], orbital_edge_gc, aggr="mean", root_weight=True)
        self.orbital_vertex_gru = nn.GRU(orbital_vertex_dim[1], orbital_vertex_dim[1])

        self.orbital_s2s = Set2Set(orbital_vertex_dim[1], processing_steps=s2s_step)
        self.orbital_lin0 = nn.Sequential(nn.Linear(orbital_vertex_dim[0], 2 * orbital_vertex_dim[0]), nn.CELU(),
                                          nn.Linear(2 * orbital_vertex_dim[0], orbital_vertex_dim[1]), nn.CELU())
        self.orbital_lin1 = nn.Sequential(nn.Linear(orbital_edge_dim[0], 2 * orbital_edge_dim[0]), nn.CELU(),
                                          nn.Linear(2 * orbital_edge_dim[0], orbital_edge_dim[1]), nn.CELU())
        self.orbital_lin2 = nn.Sequential(nn.Linear(2 * orbital_vertex_dim[1], 4 * orbital_vertex_dim[1]), nn.CELU())

        # cross net
        self.cross_lin0 = nn.Sequential(
            nn.Linear(4 * atom_vertex_dim[1] + 4 * orbital_vertex_dim[1], 4 * output_dim),
            nn.CELU(),
            nn.Linear(4 * output_dim, output_dim)
        )
        self.cross_o2a_lin = nn.Sequential(nn.Linear(orbital_vertex_dim[1], 2 * orbital_vertex_dim[1]), nn.CELU(),
                                           nn.Linear(2 * orbital_vertex_dim[1], int(atom_vertex_dim[1] / 2)), nn.CELU())
        self.cross_o2a_s2s = Set2Set(int(atom_vertex_dim[1] / 2), processing_steps=s2s_step)
        self.cross_o2a_gru = nn.GRU(atom_vertex_dim[1], atom_vertex_dim[1])
        self.cross_a2o_lin = nn.Sequential(nn.Linear(atom_vertex_dim[1], 2 * atom_vertex_dim[1]), nn.CELU(),
                                           nn.Linear(2 * atom_vertex_dim[1], orbital_vertex_dim[1]), nn.CELU())
        self.cross_a2o_gru = nn.GRU(orbital_vertex_dim[1], orbital_vertex_dim[1])

    @staticmethod
    def gen_orbital_batched_idx(orbital_data):
        orbital_batched_idx = torch.zeros_like(orbital_data.atom_idx)
        last_idx = 0
        for i in range(orbital_data.atom_idx.shape[0]):
            orbital_batched_idx[i] = last_idx + orbital_data.atom_idx[i]
            if i < orbital_data.atom_idx.shape[0] - 1 and orbital_data.batch[i] != orbital_data.batch[i + 1]:
                last_idx = orbital_batched_idx[i] + 1
        return orbital_batched_idx

    def forward(self, atom_data, orbital_data):
        orbital_batched_idx = self.gen_orbital_batched_idx(orbital_data)

        atom_vertex = self.atom_lin0(atom_data.x)
        atom_edge = self.atom_lin1(atom_data.edge_attr)
        orbital_vertex = self.orbital_lin0(orbital_data.x)
        orbital_edge = self.orbital_lin1(orbital_data.edge_attr)

        atom_vertex_h, orbital_vertex_h = atom_vertex.unsqueeze(0), orbital_vertex.unsqueeze(0)
        cross_a2o_h, cross_o2a_h = orbital_vertex.unsqueeze(0), atom_vertex.unsqueeze(0)

        for i in range(self.mp_step):
            # Electron Vertex -> Atom Vertex
            cross_atom_m = self.cross_o2a_s2s(self.cross_o2a_lin(orbital_vertex), orbital_batched_idx)
            atom_vertex, cross_o2a_h = self.cross_o2a_gru(cross_atom_m.unsqueeze(0), cross_o2a_h)
            atom_vertex = atom_vertex.squeeze(0)
            # Atom Vertex -> Electron Vertex
            cross_orbital_m = torch.index_select(self.cross_a2o_lin(atom_vertex), 0, orbital_batched_idx)
            orbital_vertex, cross_a2o_h = self.cross_a2o_gru(cross_orbital_m.unsqueeze(0), cross_a2o_h)
            orbital_vertex = orbital_vertex.squeeze(0)
            # Atom Edge -> Atom Vertex
            atom_vertex_m = nn.CELU()(self.atom_vertex_conv(atom_vertex, atom_data.edge_index, atom_edge))
            atom_vertex, atom_vertex_h = self.atom_vertex_gru(atom_vertex_m.unsqueeze(0), atom_vertex_h)
            atom_vertex = atom_vertex.squeeze(0)
            # Electron Edge -> Electron Vertex
            orbital_vertex_m = nn.CELU()(self.orbital_vertex_conv(orbital_vertex, orbital_data.edge_index, orbital_edge))
            orbital_vertex, orbital_vertex_h = self.orbital_vertex_gru(orbital_vertex_m.unsqueeze(0), orbital_vertex_h)
            orbital_vertex = orbital_vertex.squeeze(0)

        atom_out = self.atom_s2s(atom_vertex, atom_data.batch)
        atom_out = self.atom_lin2(atom_out)
        orbital_out = self.orbital_s2s(orbital_vertex, orbital_data.batch)
        orbital_out = self.orbital_lin2(orbital_out)

        # cross net
        cross_out = self.cross_lin0(torch.cat([atom_out, orbital_out], dim=1))
        return cross_out
