"""
Created on 23 July 2024

@author: kevin
"""

import torch
import time
import numpy as np
from .metrics import metrics


def full_batch_step(model, optimizer, criterion, x_train, y_train, 
                    adj_train, train_mask, logging = False, adj_cond = None):
    """
    Perform a single optimization step on the model using a full batch of training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        criterion (callable): The loss function.
        x_train (torch.Tensor): The input features for the training data.
        y_train (torch.Tensor): The target labels for the training data.
        adj_train (torch.Tensor): The adjacency matrix for the training data.
        train_mask (torch.Tensor or None): Mask indicating which nodes to consider for training.
        logging (bool, optional): Whether to log training metrics. Defaults to False.
        adj_cond (torch.Tensor or None, optional): The condensed adjacency matrix. Defaults to None.
    Returns:
        loss (torch.Tensor): The computed loss for the current batch.
    """
    model.train()
    optimizer.zero_grad()

    if adj_cond:
        out = model(x_train, adj_train, adj_cond)
    else:
        out = model(x_train, adj_train)
    if train_mask == None:
        loss = criterion(out, y_train)
    else:
        loss = criterion(out[train_mask], y_train[train_mask])
    loss.backward()
    optimizer.step()

    if logging:
        acc,micro_f1,sens,spec = metrics(out,y_train)
        print(f"Train accuracy: {acc}, Train micro_f1: {micro_f1},Train Sens: {sens}, Train Spec: {spec}")

    return loss
    

def evaluate(model, x, y, adj, mask, adj_cond = None):
    """
    Evaluate the model on the provided data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        x (torch.Tensor): The input features for the evaluation data.
        y (torch.Tensor): The target labels for the evaluation data.
        adj (torch.Tensor): The adjacency matrix for the evaluation data.
        mask (torch.Tensor): Mask indicating which nodes to consider for evaluation.
        adj_cond (torch.Tensor or None, optional): The condensed adjacency matrix. Defaults to None.

    Returns:
        acc (float): The accuracy of the model on the evaluation data.
        micro_f1 (float): The micro-averaged F1 score on the evaluation data.
        sens (float): The sensitivity (recall) of the model on the evaluation data.
        spec (float): The specificity of the model on the evaluation data.
    """
    with torch.no_grad():
        model.eval()
        if adj_cond:
            out = model(x, adj, adj_cond).squeeze()
        else:
            out = model(x, adj).squeeze()
        acc,micro_f1,sens,spec = metrics(out[mask],y[mask])
    
    return acc, micro_f1, sens, spec


def train(model, device, x_train, y_train, adj_train, adj_train_cond = None, train_mask = None, x_val = None, 
          y_val = None, adj_val = None, adj_val_cond = None, val_mask = None, x_test = None, 
          y_test = None, adj_test = None, adj_test_cond = None, test_mask = None, multilabel = True, 
          lr = 0.0005, num_epoch = 100):
    """
    Train a model using the provided training data and evaluate it on validation and test data.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        x_train (torch.Tensor): The input features for the training data.
        y_train (torch.Tensor): The target labels for the training data.
        adj_train (torch.Tensor): The adjacency matrix for the training data.
        adj_train_cond (torch.Tensor or None, optional): The condensed adjacency matrix for training. Defaults to None.
        train_mask (torch.Tensor or None, optional): Mask indicating which nodes to consider for training. Defaults to None.
        x_val (torch.Tensor or None, optional): The input features for the validation data. Defaults to None.
        y_val (torch.Tensor or None, optional): The target labels for the validation data. Defaults to None.
        adj_val (torch.Tensor or None, optional): The adjacency matrix for the validation data. Defaults to None.
        adj_val_cond (torch.Tensor or None, optional): The condensed adjacency matrix for validation. Defaults to None.
        val_mask (torch.Tensor or None, optional): Mask indicating which nodes to consider for validation. Defaults to None.
        x_test (torch.Tensor or None, optional): The input features for the test data. Defaults to None.
        y_test (torch.Tensor or None, optional): The target labels for the test data. Defaults to None.
        adj_test (torch.Tensor or None, optional): The adjacency matrix for the test data. Defaults to None.
        adj_test_cond (torch.Tensor or None, optional): The condensed adjacency matrix for testing. Defaults to None.
        test_mask (torch.Tensor or None, optional): Mask indicating which nodes to consider for testing. Defaults to None.
        multilabel (bool, optional): Whether the task is multilabel classification. Defaults to True.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.0005.
        num_epoch (int, optional): Number of training epochs. Defaults to 100.
    Returns:
        tuple: A tuple containing:
            - max_val_acc (float): Best validation accuracy achieved.
            - max_val_f1 (float): Best validation F1 score achieved.
            - max_val_sens (float): Best validation sensitivity achieved.
            - max_val_spec (float): Best validation specificity achieved.
            - max_val_test_acc (float): Best test accuracy achieved.
            - max_val_test_f1 (float): Best test F1 score achieved.
            - max_val_test_sens (float): Best test sensitivity achieved.
            - max_val_test_spec (float): Best test specificity achieved.
            - session_memory (float): Peak GPU memory usage during the session (in MB).
            - train_memory (float): Peak GPU memory usage during training (in MB).
            - train_time_avg (float): Average time per epoch during training.
    """

    # passing model and training data to GPU
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    adj_train = adj_train.to(device)
    if adj_train_cond:
        adj_train_cond = adj_train_cond.to(device)
    if train_mask != None:
        train_mask = train_mask.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    max_val_acc = 0
    max_val_sens = 0
    max_val_spec = 0
    max_val_f1 = 0
    max_val_test_acc = 0
    max_val_test_sens = 0
    max_val_test_spec = 0
    max_val_test_f1 = 0
    
    time_arr = np.zeros((num_epoch,))

    for epoch in range(num_epoch):
            
        # single mini batch step
        t = time.time()

        loss = full_batch_step(model, optimizer, criterion, 
                                   x_train, y_train, adj_train, train_mask, 
                                   logging=False, adj_cond=adj_train_cond)
        
        time_per_epoch = time.time() - t
        time_arr[epoch] = time_per_epoch
        
        if epoch == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
            
            # passing validation and test data to GPU (we do it after first forward pass to get)
            # accurate pure training GPU memory usage
            if x_val != None and y_val != None and adj_val != None and val_mask != None:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                adj_val = adj_val.to(device)
                val_mask = val_mask.to(device)
                if adj_val_cond:
                    adj_val_cond = adj_val_cond.to(device)
                if x_test != None and y_test != None and adj_test != None and test_mask != None:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    adj_test = adj_test.to(device)
                    test_mask = test_mask.to(device)
                    if adj_test_cond:
                        adj_test_cond = adj_test_cond.to(device)
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}')
            print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
        
        # evaluation
        if x_val != None and y_val != None:
            acc, micro_f1, sens, spec = evaluate(model, x_val, y_val, adj_val, 
                                                 val_mask, adj_cond = adj_val_cond)
            
            if epoch % 100 == 0:
                print(f"Val accuracy: {acc}, Val micro_f1: {micro_f1}, Val Sens: {sens}, Val Spec: {spec}")
            
            if acc > max_val_acc:
                max_val_acc = acc
                max_val_f1 = micro_f1
                max_val_sens = sens
                max_val_spec = spec
                
                if (x_test != None and y_test != None):
                    acc, micro_f1, sens, spec = evaluate(model, x_test, y_test, 
                                                         adj_test, test_mask, adj_cond = adj_test_cond)
                    max_val_test_acc = acc
                    max_val_test_f1 = micro_f1
                    max_val_test_sens = sens
                    max_val_test_spec = spec
                    
                    print("===========================================Best Model Update:=======================================")
                    print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
                    print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
                    print("====================================================================================================")

    print("Best Model:")
    print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
    print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
    print(f"Average time per epoch: {time_arr[10:].mean()}") # don't include the first few epoch (slower due to Torch initialization)
    print(f"Training GPU Memory Usage: {train_memory} MB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
    
    # cleaning memory and stats
    session_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
    train_time_avg = time_arr[10:].mean()
    del x_val
    del y_val
    del x_test
    del y_test
    model = model.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    return (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc,
            max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
            train_memory, train_time_avg)
