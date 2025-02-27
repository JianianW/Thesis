import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', bias=True, **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Calculate normalization: D^{-1/2} * (A + I) * D^{-1/2}
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)  # Degree matrix D
        deg_inv_sqrt = deg.pow(-0.5)  # D^{-1/2}
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle inf values

        # Compute normalization for each edge: D^{-1/2} * A * D^{-1/2}
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Apply the GCN layer: h = (D^{-1/2} A D^{-1/2}) * x * W
        h = torch.matmul(x, self.weight)
        aggr_out = self.propagate(edge_index, x=h, norm=norm)
        return aggr_out + self.lin(x)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
