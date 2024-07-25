"""
Created on 23 July 2024

@author: kevin
"""

import torch
import torch.nn.functional as F

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm=False, dropout=0.0, 
                 drop_input=False, residual=False, graph_task=False):
        """
        Initialize the Graph Isomorphism Network (GIN) model.

        Args:
            hidden_channels (int): Number of hidden channels in MLP layers.
            num_layers (int): Number of GIN layers in the network.
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
        
        self.mlp_layers = torch.nn.ModuleList()  # List to store MLP layers
        self.batch_norm_layers = torch.nn.ModuleList()  # List to store batch normalization layers
        
        # Adding input layer MLP
        if residual:
            self.mlp_layers.append(torch.nn.Sequential(
                torch.nn.Linear(2 * in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ))
        else:
            self.mlp_layers.append(torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Adding hidden layer MLPs
        for i in range(num_layers - 2):
            if residual:
                self.mlp_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                ))
            else:
                self.mlp_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                ))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding output layer MLP
        if residual:
            self.mlp_layers.append(torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, out_channels)
            ))
        else:
            self.mlp_layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, out_channels)
            ))

    def forward(self, x, adj):
        """
        Forward pass of the GIN model.

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
            # Aggregation phase: Compute graph convolution and add residual connections
            if self.residual:
                x = torch.cat((adj @ x + x, x), 1)
            else:
                x = adj @ x + x
            
            # Transformation phase: Apply MLP layer and optional batch normalization
            x = self.mlp_layers[i](x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = F.relu(x)  # Activation function
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        
        # Aggregation phase for the output layer
        if self.residual:
            x = torch.cat((adj @ x + x, x), 1)
        else:
            x = adj @ x + x
        
        # Transformation phase for the output layer
        x = self.mlp_layers[-1](x)

        # Graph-level readout if graph_task is True
        if self.graph_task:
            x = torch.mean(x, dim=0)  # Aggregate node features to a single graph-level feature
        
        return x
