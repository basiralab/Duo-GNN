"""
Created on 23 July 2024

@author: kevin
"""

import torch
import torch.nn.functional as F

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm=False, dropout=0.0, 
                 drop_input=False, residual=False, graph_task=False):
        """
        Initialize the Graph Convolutional Network (GCN) model.

        Args:
            hidden_channels (int): Number of hidden channels in GCN layers.
            num_layers (int): Number of GCN layers in the network.
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
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
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        self.residual = residual
        self.graph_task = graph_task
        
        self.linear_layers = torch.nn.ModuleList()  # List to store linear layers
        self.batch_norm_layers = torch.nn.ModuleList()  # List to store batch normalization layers
        
        # Adding input layer
        if residual:
            self.linear_layers.append(torch.nn.Linear(2 * in_channels, hidden_channels))
        else:
            self.linear_layers.append(torch.nn.Linear(in_channels, hidden_channels))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Adding hidden layers
        for i in range(num_layers - 2):
            if residual:
                self.linear_layers.append(torch.nn.Linear(2 * hidden_channels, hidden_channels))
            else:
                self.linear_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding output layer
        if residual:
            self.linear_layers.append(torch.nn.Linear(2 * hidden_channels, out_channels))
        else:
            self.linear_layers.append(torch.nn.Linear(hidden_channels, out_channels))

        # Graph-level readout layer if graph_task is True
        if graph_task:
            self.graph_readout = torch.nn.Linear(out_channels, 1)

    def forward(self, x, adj):
        """
        Forward pass of the GCN model.

        Args:
            x (Tensor): Input feature matrix.
            adj (Tensor): Adjacency matrix for graph convolution.

        Returns:
            Tensor: Output feature matrix or graph-level output based on graph_task.
        """
        # Apply dropout to input features if drop_input is set to True
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(self.num_layers - 1):  # Exclude output layer
            # Aggregation phase: Apply graph convolution
            if self.residual:
                if x.dim() == 3:
                    # If 3D tensor, concatenate along feature dimension
                    x = torch.cat((adj @ x, x), dim=1)
                else:
                    x = torch.cat((adj @ x, x), dim=1)
            else:
                x = adj @ x
            
            # Transformation phase: Apply linear transformation and batch normalization
            x = self.linear_layers[i](x)
            if self.batch_norm:
                if x.dim() == 3:
                    # Swap the second and third dimensions for batch normalization
                    x = x.transpose(1, 2)
                x = self.batch_norm_layers[i](x)
                if x.dim() == 3:
                    # Swap dimensions back after batch normalization
                    x = x.transpose(1, 2)

            # Apply activation function and dropout
            if self.graph_task:
                x = x.tanh()  # Use tanh for graph-level tasks
            else:
                x = x.relu()  # Use ReLU for node-level tasks
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Aggregation phase for the output layer
        if self.residual:
            if x.dim() == 3:
                x = torch.cat((adj @ x, x), dim=1)
            else:
                x = torch.cat((adj @ x, x), dim=1)
        else:
            x = adj @ x
        
        # Transformation phase for the output layer
        x = self.linear_layers[-1](x)

        # Graph-level readout if graph_task is True
        if self.graph_task:
            if x.dim() == 3:
                x = x.transpose(1, 2)  # Swap dimensions before readout
                x = self.graph_readout(x)
                x = x.squeeze(dim=2)  # Remove singleton dimension
            else:
                x = self.graph_readout(x)
        
        return x
