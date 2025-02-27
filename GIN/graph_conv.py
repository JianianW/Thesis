import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class GINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_mlp_layers=2, learn_eps=False):
        super(GINConv, self).__init__(aggr='add')  # GIN uses 'add' aggregation
        self.mlp = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            *[Sequential(Linear(out_channels, out_channels), ReLU()) for _ in range(num_mlp_layers - 1)],
        )
        self.epsilon = torch.nn.Parameter(torch.zeros(1)) if learn_eps else 0

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.mlp((1 + self.epsilon) * x + self.propagate(edge_index, x=x))

    def message(self, x_j):
        return x_j
