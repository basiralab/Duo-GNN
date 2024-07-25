"""
Created on 23 July 2024

@author: kevin
"""

import numpy as np
from sklearn import metrics as sk
import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_medmnist_data(view, split):
    """
    Load MedMNIST data for the given view and split.

    Args:
        view (str): Either 'c' or 's' indicating the dataset view.
        split (str): The dataset split, one of 'train', 'val', or 'test'.

    Returns:
        (np.ndarray, np.ndarray): Tuple of images and labels.
    """
    info = INFO[f'organ{view}mnist']
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DataClass(split=split, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data in dataloader:
        images, labels = data
    images = images.numpy().squeeze()
    labels = labels.numpy().squeeze()
    return images, labels

def process_data(sparsity):
    """
    Process and save the Organ dataset in both sparse and dense formats.

    Args:
        sparsity (bool): Whether to process the dataset as sparse or dense.
    """
    views = ['c', 's']
    num_features = 28 * 28

    for view in views:
        # Load training data
        train_images, train_labels = load_medmnist_data(view, 'train')
        train_data = train_images.reshape(train_images.shape[0], num_features).astype('float')
        num_train = train_images.shape[0]

        # Load validation data
        val_images, val_labels = load_medmnist_data(view, 'val')
        val_data = val_images.reshape(val_images.shape[0], num_features).astype('float')
        num_val = val_images.shape[0]

        # Load test data
        test_images, test_labels = load_medmnist_data(view, 'test')
        test_data = test_images.reshape(test_images.shape[0], num_features).astype('float')
        num_test = test_images.shape[0]

        # Concatenate train, validation, and test data
        num_data = num_train + num_val + num_test
        data_feat = np.concatenate((train_data, val_data, test_data), axis=0)
        data_label = np.concatenate((train_labels, val_labels, test_labels), axis=0).reshape(-1)

        # Construct and scale adjacency matrix
        adj_matrix = sk.pairwise.cosine_similarity(data_feat, data_feat)
        adj_matrix = (adj_matrix - adj_matrix.min()) / (adj_matrix.max() - adj_matrix.min())

        # Apply sparsity thresholds
        if view == 'c':  # Organ-C
            threshold = 0.972 if sparsity else 0.965
        elif view == 's':  # Organ-S
            threshold = 0.977 if sparsity else 0.970
        
        adj_matrix = adj_matrix > threshold

        # Generate masks
        train_mask = np.zeros(num_data, dtype=bool)
        train_mask[:num_train] = True
        val_mask = np.zeros(num_data, dtype=bool)
        val_mask[num_train:num_train + num_val] = True
        test_mask = np.zeros(num_data, dtype=bool)
        test_mask[num_train + num_val:] = True

        # Save masks, features, labels, and edge index
        suffix = '_sparse' if sparsity else '_dense'
        base_path = f"organ{view}{suffix}"

        np.save(f"{base_path}/train_mask.npy", train_mask)
        np.save(f"{base_path}/val_mask.npy", val_mask)
        np.save(f"{base_path}/test_mask.npy", test_mask)
        np.save(f"{base_path}/data_feat.npy", data_feat)
        np.save(f"{base_path}/data_label.npy", data_label)

        # Generate and save edge index
        edge_index = np.array([[i, j] for i in range(num_data) for j in range(num_data) if i != j and adj_matrix[i, j]])
        np.save(f"{base_path}/edge_index.npy", edge_index)

        print(f"View-{view.upper()} ({'Sparse' if sparsity else 'Dense'}) generated!")

# Call the function with sparsity parameter
process_data(sparsity=True)
process_data(sparsity=False)
