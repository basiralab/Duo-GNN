"""
Created on 23 July 2024

@author: kevin
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

"""
    GAT: Graph Attention Networks
    Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
    Graph Attention Networks (ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, heads=1, batch_norm=False, 
                 dropout=0.0, drop_input=False, residual=False, graph_task=False):
        """
        Initialize the Graph Attention Network (GAT) model.

        Args:
            hidden_channels (int): Number of hidden channels in GAT layers.
            num_layers (int): Number of GAT layers in the network.
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            heads (int, optional): Number of attention heads. Default is 1.
            batch_norm (bool, optional): Whether to apply batch normalization. Default is False.
            dropout (float, optional): Dropout rate for dropout layers. Default is 0.0.
            drop_input (bool, optional): Whether to apply dropout to input features. Default is False.
            residual (bool, optional): Whether to use residual connections. Default is False.
            graph_task (bool, optional): Whether the task is a graph-level task. Default is False.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        self.residual = residual
        self.graph_task = graph_task
        
        self.gat_layers = torch.nn.ModuleList()  # List to store GAT layers
        self.batch_norm_layers = torch.nn.ModuleList()  # List to store batch normalization layers
        
        # Adding input layer
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels * heads))
            
        # Adding hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        
        # Adding output layer
        self.gat_layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        
        # Graph-level readout layer if graph_task is True
        if graph_task:
            self.graph_readout = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        Args:
            x (Tensor): Input feature matrix.
            edge_index (LongTensor): Edge indices in COO format.

        Returns:
            Tensor: Output feature matrix or graph-level output based on graph_task.
        """
        # Apply dropout to input features if drop_input is set to True
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Process through hidden GAT layers
        for i in range(self.num_layers - 1):  # Exclude the output layer
            x = self.gat_layers[i](x, edge_index)
            
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            
            # Activation function and dropout
            if self.graph_task:
                x = x.tanh()  # Use tanh for graph-level tasks
            else:
                x = x.relu()  # Use ReLU for node-level tasks
                
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.gat_layers[-1](x, edge_index)
        
        # Graph-level readout if graph_task is True
        if self.graph_task:
            x = self.graph_readout(x)
        
        return x

