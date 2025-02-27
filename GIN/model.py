import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from graph_conv import GINConv

class Net(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps):
        super(Net, self).__init__()
        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # Define GIN convolution layers
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers - 1):
            if layer == 0:
                self.gin_layers.append(GINConv(input_dim, hidden_dim, num_mlp_layers, learn_eps))
            else:
                self.gin_layers.append(GINConv(hidden_dim, hidden_dim, num_mlp_layers, learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear prediction layer: for concatenated embeddings
        self.linears_prediction = nn.ModuleList([
            nn.Linear(input_dim + (num_layers - 1) * hidden_dim, output_dim)
        ])

    def reset_parameters(self):
        for layer in self.gin_layers:
            layer.mlp.apply(self._init_weights)
        for bn in self.batch_norms:
            bn.reset_parameters()
        for linear in self.linears_prediction:
            linear.reset_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_graph_embedding(self, x, edge_index, batch):
        """Generate concatenated graph-level embeddings for all layers."""
        hidden_rep = [global_add_pool(x, batch)]
        h = x

        for layer in range(self.num_layers - 1):
            h = self.gin_layers[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.final_dropout, training=self.training)
            hidden_rep.append(global_add_pool(h, batch))

        graph_embedding = torch.cat(hidden_rep, dim=1)
        return graph_embedding

    def forward(self, x, edge_index, batch):
        """Final prediction using concatenated embeddings."""
        graph_embedding = self.get_graph_embedding(x, edge_index, batch)
        output = self.linears_prediction[0](graph_embedding)
        return F.dropout(output, self.final_dropout, training=self.training)
