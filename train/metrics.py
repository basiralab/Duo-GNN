"""
Created on 23 July 2024

@author: kevin
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import networkx as nx


def calculate_homophily_ratios(adj, x, y):
    """
    Calculate homophily ratios based on feature and label similarities.

    Args:
        adj (scipy.sparse.coo_matrix): Adjacency matrix in COO format.
        x (numpy.ndarray or torch.Tensor): Node features.
        y (numpy.ndarray or torch.Tensor): Node labels.

    Returns:
        list: List of homophily ratios for each edge in the adjacency matrix.
    """
    homophily_ratios = []
    
    # Extract COO format components
    row, col, _ = adj.coo()
    
    for r, c in zip(row.tolist(), col.tolist()):
        # Extract features and labels using row and col indices
        x1, y1 = x[r], y[r]
        x2, y2 = x[c], y[c]
        
        # Calculate similarity in features (assuming features are numpy arrays)
        feature_similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        
        # Calculate similarity in labels
        label_similarity = 1 if y1 == y2 else 0
        
        # Homophily ratio can be a combination of both similarities
        homophily_ratio = 0.5 * feature_similarity + 0.5 * label_similarity
        
        homophily_ratios.append(homophily_ratio)
    
    return homophily_ratios

def logit_to_label(out):
    """
    Convert logits to predicted labels using argmax.

    Args:
        out (torch.Tensor): Logits tensor.

    Returns:
        torch.Tensor: Predicted labels.
    """
    return out.argmax(dim=1)


def metrics(logits, y):
    """
    Calculate classification metrics.

    Args:
        logits (torch.Tensor): Model output logits.
        y (torch.Tensor): True labels.

    Returns:
        tuple: (accuracy, micro F1 score, sensitivity, specificity)
    """
    if y.dim() == 1: # Multi-class
        y_pred = logit_to_label(logits)
        cm = confusion_matrix(y.cpu(),y_pred.cpu())
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
    
        acc = np.diag(cm).sum() / cm.sum()
        micro_f1 = acc # micro f1 = accuracy for multi-class
        sens = TP.sum() / (TP.sum() + FN.sum())
        spec = TN.sum() / (TN.sum() + FP.sum())
    
    else: # Multi-label
        y_pred = logits >= 0
        y_true = y >= 0.5
        
        tp = int((y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        
        acc = (tp + tn)/(tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        sens = (tp)/(tp + fn)
        spec = (tn)/(tn + fp)
        
    return acc, micro_f1, sens, spec

def edge_degree_centrality(graph):
    """
    Compute the degree centrality for each edge in the graph.

    Args:
        graph (networkx.Graph): A NetworkX graph object.

    Returns:
        dict: A dictionary where keys are edges and values are the average degree of the nodes incident to the edge.
    """
    edge_degree = {}
    
    # Iterate over edges
    for edge in graph.edges():
        u, v = edge
        
        # Compute degrees of the nodes incident to the edge
        degree_u = graph.degree(u)
        degree_v = graph.degree(v)
        
        # Calculate average degree
        avg_degree = (degree_u + degree_v) / 2
        
        # Assign average degree as edge degree centrality
        edge_degree[edge] = avg_degree
    
    return edge_degree

def edge_metric_compute(metric, graph):
    """
    Compute a specified edge metric by averaging the provided node metrics for each edge.

    Args:
        metric (dict): A dictionary where keys are nodes and values are the metric values for the nodes.
        graph (networkx.Graph): A NetworkX graph object.

    Returns:
        dict: A dictionary where keys are edges and values are the average metric values of the nodes incident to the edge.
    """
    edges = {}
    
    # Iterate over edges
    for edge in graph.edges():
        u, v = edge
        
        metric_u = metric[u]
        metric_v = metric[v]
        
        avg_centrality = (metric_u + metric_v) / 2
        
        edges[edge] = avg_centrality
    
    return edges
