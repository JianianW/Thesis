import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from graph_conv import GraphConv  # Import GraphConv from graph_conv.py

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(Net, self).__init__()
        # Define three graph convolution layers
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

        # MLP layers after pooling
        self.mlp_hidden = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mlp_output = torch.nn.Linear(hidden_channels, out_channel)

    # Reset parameters for each layer to ensure independent experiments
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.mlp_hidden.reset_parameters()
        self.mlp_output.reset_parameters()

    def forward(self, x, edge_index, batch):
        # Standard forward pass process
        x = self.get_graph_embedding(x, edge_index, batch)
        
        # Pass through MLP layers
        x = self.mlp_forward(x)
        return x

    def get_graph_embedding(self, x, edge_index, batch):
        # Obtain the embedding before MLP layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global add pooling layer
        x = global_add_pool(x, batch)
        
        return x

    def mlp_forward(self, x):
        # Pass the graph embedding through MLP layers
        x = F.relu(self.mlp_hidden(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp_output(x)
        return x
