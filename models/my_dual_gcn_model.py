"""
Created on 23 July 2024

@author: kevin
"""

import torch
import torch.nn.functional as F

class DualGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm=False, dropout=0.0, 
                 drop_input=False, residual=False, graph_task=False, max_communities=500):
        """
        Initializes the DualGCN model.

        Args:
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of layers.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            batch_norm (bool): Whether to use batch normalization.
            dropout (float): Dropout rate.
            drop_input (bool): Whether to apply dropout to input.
            residual (bool): Whether to use residual connections.
            graph_task (bool): Whether the task is graph-level.
            max_communities (int): Maximum number of communities.
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
        self.max_communities = max_communities
        
        self.linear_layers_a = torch.nn.ModuleList()
        self.linear_layers_b = torch.nn.ModuleList()
        self.batch_norm_layers_a = torch.nn.ModuleList()
        self.batch_norm_layers_b = torch.nn.ModuleList()

        # Adding input layer
        input_dim = 2 * in_channels if residual else in_channels
        self.linear_layers_a.append(torch.nn.Linear(input_dim, hidden_channels))
        self.linear_layers_b.append(torch.nn.Linear(input_dim, hidden_channels))
        
        if self.batch_norm:
            self.batch_norm_layers_a.append(torch.nn.BatchNorm1d(hidden_channels))
            self.batch_norm_layers_b.append(torch.nn.BatchNorm1d(hidden_channels))

        # Adding hidden layers
        for _ in range(num_layers - 2):
            hidden_dim = 2 * hidden_channels if residual else hidden_channels
            self.linear_layers_a.append(torch.nn.Linear(hidden_dim, hidden_channels))
            self.linear_layers_b.append(torch.nn.Linear(hidden_dim, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers_a.append(torch.nn.BatchNorm1d(hidden_channels))
                self.batch_norm_layers_b.append(torch.nn.BatchNorm1d(hidden_channels))

        self.merge_layer_b = torch.nn.Linear(num_layers * hidden_dim, hidden_channels)

        # Adding output layer
        output_dim = 4 * hidden_channels if residual else 2 * hidden_channels
        self.output_layer = torch.nn.Linear(output_dim, out_channels)

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initializes the parameters of the linear layers."""
        for layer in self.linear_layers_a:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)
        for layer in self.linear_layers_b:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    def forward_single(self, x, adj, heterophily=False):
        """
        Forward pass for a single graph.

        Args:
            x (Tensor): Input features.
            adj (SparseTensor): Adjacency matrix.
            heterophily (bool): Whether to use heterophily graph.

        Returns:
            Tensor: Output features.
        """
        if heterophily:
            all_x = []

        for i in range(self.num_layers - 1):  # exclude output layer
            next_x = adj @ x  # Compute next_x first

            # aggregation phase
            if self.residual:
                if i == 0 and heterophily:
                    x = torch.cat((x, x), 1)
                else:
                    x = torch.cat((next_x, x), 1)
            else:
                if i != 0:
                    x = next_x

            # transformation phase
            if not heterophily:
                x = self.linear_layers_a[i](x)
                if self.batch_norm:
                    x = self.batch_norm_layers_a[i](x)
            else:
                x = self.linear_layers_b[i](x)
                if self.batch_norm:
                    x = self.batch_norm_layers_b[i](x)
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if heterophily:
                all_x.append(x)

        # aggregation phase (output layer)
        next_x = adj @ x  # Compute next_x first

        if self.residual:
            x = torch.cat((next_x, x), 1)
        else:
            x = next_x

        if heterophily:
            all_x.append(x)
            x = self.merge_layer_b(torch.cat(all_x, dim=1))
            x = x.relu_()

        if self.graph_task:
            x = torch.mean(x, dim=0)

        return x

    def forward(self, x, adj_a, adj_b):
        """
        Forward pass through the DualGCN model.

        Args:
            x (Tensor): Input features.
            adj_a (SparseTensor): Adjacency matrix for the first graph (homogeneous).
            device (torch.device): Device to use (CPU or GPU).
            adj_b (SparseTensor, optional): Adjacency matrix for the second graph (heterogeneous).

        Returns:
            Tensor: Output features.
        """
        # if using input dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_a = x.clone()
        x_b = x.clone()

        x_a = self.forward_single(x_a,adj_a, heterophily = False)

        x_b = self.forward_single(x_b, adj_b, heterophily = True)
        
        x = torch.cat((x_a, x_b), 1)
        
        x = self.output_layer(x)

        return x
