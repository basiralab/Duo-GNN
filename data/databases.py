"""
Created on 23 July 2024

@author: kevin
"""

import torch
import numpy as np
import torch.utils.data
import sklearn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split

def get_planetoid_dataset(name="Cora"):
    """
    Load the Planetoid dataset (Cora, CiteSeer, or PubMed) with normalized features.

    Args:
        name (str): The name of the dataset. Must be one of 'Cora', 'CiteSeer', or 'PubMed'.

    Returns:
        tuple: A tuple containing node features, labels, edge indices, 
               and masks for training, validation, and testing.
    """
    assert name in ["Cora", "CiteSeer", "PubMed"], "Dataset name must be 'Cora', 'CiteSeer', or 'PubMed'."
    dataset = Planetoid(root=f'/tmp/{name}', name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    x, y, edge_index = data.x, data.y, data.edge_index
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    return x, y, edge_index, train_mask, val_mask, test_mask

def get_organ_dataset(view='C', sparse=True, balanced=True):
    """
    Load the Organ dataset (C or S view) in either sparse or dense format.

    Args:
        view (str): The view of the dataset. Must be 'C' or 'S'.
        sparse (bool): Whether to load the sparse version of the dataset.

    Returns:
        tuple: A tuple containing node features, labels, edge indices, 
               and masks for training, validation, and testing.
    """
    print(f"Loading Organ-{view} Dataset...")
    dataset_name = f'organ{view.lower()}{"_sparse" if sparse else "_dense"}'
    
    # Load masks
    train_mask = torch.tensor(np.load(f'data/{dataset_name}/train_mask.npy'))
    val_mask = torch.tensor(np.load(f'data/{dataset_name}/val_mask.npy'))
    test_mask = torch.tensor(np.load(f'data/{dataset_name}/test_mask.npy'))

    # Load labels
    labels = np.load(f'data/{dataset_name}/data_label.npy')

    # Load and normalize features
    features = np.load(f'data/{dataset_name}/data_feat.npy')
    features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    # Load edge indices
    edge_index = np.load(f'data/{dataset_name}/edge_index.npy')
    
    # Print dataset statistics
    print_statistics(features, labels, edge_index, train_mask, val_mask, test_mask)

    if not balanced:
        all_labels = [0,1,2,3,4,5,6,7,8,9,10]
        chosen_labels = [0, 1, 2, 4]

        print("[Before unbalancing] Class distribution in the training set:")
        for label in all_labels:
                count = np.sum(labels[train_mask] == label)
                print(f"Label {label}: {count} samples")
        print("[Before unbalancing] Class distribution in the validation set:")
        for label in all_labels:
                count = np.sum(labels[val_mask] == label)
                print(f"Label {label}: {count} samples")

        print("[Before unbalancing] Class distribution in the test set:")
        for label in all_labels:
                count = np.sum(labels[test_mask] == label)
                print(f"Label {label}: {count} samples")

        chosen_indices = np.where(np.isin(labels[train_mask], chosen_labels))[0]
        train_indices, test_indices = train_test_split(chosen_indices, test_size=0.8, stratify=labels[train_mask][chosen_indices])
                
        new_train_mask = torch.full_like(train_mask, False)
        new_train_mask[train_indices] = True

        for i, label in enumerate(labels):
                if label in chosen_labels and new_train_mask[i] == False:
                        train_mask[i] = False

        train_mask[train_indices] = True
        test_mask[test_indices] = True

        print("Class distribution in the training set:")
        for label in all_labels:
                count = np.sum(labels[train_mask] == label)
                print(f"Label {label}: {count} samples")

        print("Class distribution in the validation set:")
        for label in all_labels:
                count = np.sum(labels[val_mask] == label)
                print(f"Label {label}: {count} samples")

        print("Class distribution in the test set:")
        for label in all_labels:
                count = np.sum(labels[test_mask] == label)
                print(f"Label {label}: {count} samples")
    
    return features, labels, edge_index, train_mask, val_mask, test_mask

def print_statistics(features, labels, edge_index, train_mask, val_mask, test_mask):
    """
    Print statistics of the dataset.

    Args:
        features (np.ndarray): Node features.
        labels (np.ndarray): Node labels.
        edge_index (np.ndarray): Edge indices.
        train_mask (torch.Tensor): Mask for training nodes.
        val_mask (torch.Tensor): Mask for validation nodes.
        test_mask (torch.Tensor): Mask for test nodes.
    """
    print("=============== Dataset Properties ===============")
    print(f"Total Nodes: {features.shape[0]}")
    print(f"Total Edges: {edge_index.shape[0]}")
    print(f"Number of Features: {features.shape[1]}")
    if labels.ndim == 1:
        print(f"Number of Labels: {labels.max() + 1}")
        print("Task Type: Multi-class Classification")
    else:
        print(f"Number of Labels: {labels.shape[1]}")
        print("Task Type: Multi-label Classification")
    print(f"Training Nodes: {train_mask.sum().item()}")
    print(f"Validation Nodes: {val_mask.sum().item()}")
    print(f"Testing Nodes: {test_mask.sum().item()}")
    print()
