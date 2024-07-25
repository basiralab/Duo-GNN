"""
Created on 23 July 2024

@author: Kevin
"""

import os

def create_empty_file_label_dist(filename):
    """
    Creates an empty file for storing label distribution statistics.
    
    Args:
        filename (str): The name of the file to create.
    """
    with open(filename, 'w') as f:
        f.write('dataset,contraction,node_num,label_dist_avg_error,label_dist_error\n')

def create_empty_file_result(filename):
    """
    Creates an empty file for storing model results with a header row.
    
    Args:
        filename (str): The name of the file to create.
    """
    with open(filename, 'w') as f:
        f.write("Dataset,Model_Type,Epochs,Node_Num,Session_Memory,Train_Memory,Train_Time_Avg,Max_Val_Acc,"
                "Max_Val_F1,Max_Val_Sens,Max_Val_Spec,Max_Val_Test_Acc,Max_Val_Test_F1,Max_Val_Test_Sens,"
                "Max_Val_Test_Spec,Hidden_Dimension,Max_Communities,Remove_Edges,Topological_Measure,"
                "Make_Unbalanced,Dense\n")

def write_result(dataset, model_type, epochs, node_num, max_val_acc, max_val_f1, 
                 max_val_sens, max_val_spec, max_val_test_acc, max_val_test_f1, 
                 max_val_test_sens, max_val_test_spec, session_memory, train_memory, 
                 train_time_avg, filename="result.csv", hidden_dimension=None, max_communities=None, 
                 remove_edges=None, topological_measure=None, make_unbalanced=None, dense=None):
    """
    Appends results to a CSV file for a specific experiment.

    Args:
        dataset (str): Name of the dataset.
        model_type (str): Type of the model used.
        epochs (int): Number of epochs for training.
        node_num (int): Number of nodes in the graph.
        max_val_acc (float): Maximum validation accuracy achieved.
        max_val_f1 (float): Maximum validation F1 score achieved.
        max_val_sens (float): Maximum validation sensitivity achieved.
        max_val_spec (float): Maximum validation specificity achieved.
        max_val_test_acc (float): Maximum test accuracy achieved.
        max_val_test_f1 (float): Maximum test F1 score achieved.
        max_val_test_sens (float): Maximum test sensitivity achieved.
        max_val_test_spec (float): Maximum test specificity achieved.
        session_memory (float): Peak memory usage during the session (in MB).
        train_memory (float): Memory usage during training (in MB).
        train_time_avg (float): Average training time per epoch (in seconds).
        filename (str): The name of the file to append results to. Default is "result.csv".
        hidden_dimension (int or None): Dimension of hidden layers, if applicable.
        max_communities (int or None): Maximum number of communities, if applicable.
        remove_edges (int or None): Number of edges removed, if applicable.
        topological_measure (str or None): Topological measure used, if applicable.
        make_unbalanced (bool or None): Indicates if the dataset is made unbalanced, if applicable.
        dense (bool or None): Indicates if the graph is dense, if applicable.
    """
    filename = "results/stats/" + filename

    # Create file if it does not exist
    if not os.path.isfile(filename):
        create_empty_file_result(filename)
    
    with open(filename, 'a') as f:
        # Write results to the file
        f.write(f"{dataset},")
        f.write(f"{model_type},")
        f.write(f"{epochs},")
        f.write(f"{node_num},")
        f.write(f"{session_memory},")
        f.write(f"{train_memory},")
        f.write(f"{train_time_avg},")
        f.write(f"{max_val_acc},")
        f.write(f"{max_val_f1},")
        f.write(f"{max_val_sens},")
        f.write(f"{max_val_spec},")
        f.write(f"{max_val_test_acc},")
        f.write(f"{max_val_test_f1},")
        f.write(f"{max_val_test_sens},")
        f.write(f"{max_val_test_spec},")
        f.write(f"{hidden_dimension if hidden_dimension is not None else '-'},")
        f.write(f"{max_communities if max_communities is not None else '-'},")
        f.write(f"{remove_edges if remove_edges is not None else '-'},")
        f.write(f"{topological_measure if topological_measure is not None else '-'},")
        f.write(f"{make_unbalanced if make_unbalanced is not None else '-'},")
        f.write(f"{dense if dense is not None else '-'}\n")

def write_result_regression(dataset, model_type, epochs, node_num, max_val_acc, max_val_f1, 
                             max_val_sens, max_val_spec, max_val_test_acc, max_val_test_f1, 
                             max_val_test_sens, max_val_test_spec, session_memory, train_memory, 
                             train_time_avg, filename="result.csv", hidden_dimension=None, 
                             max_communities=None, remove_edges=None):
    """
    Appends results for regression experiments to a CSV file.

    Args:
        dataset (str): Name of the dataset.
        model_type (str): Type of the model used.
        epochs (int): Number of epochs for training.
        node_num (int): Number of nodes in the graph.
        max_val_acc (float): Maximum validation accuracy achieved.
        max_val_f1 (float): Maximum validation F1 score achieved.
        max_val_sens (float): Maximum validation sensitivity achieved.
        max_val_spec (float): Maximum validation specificity achieved.
        max_val_test_acc (float): Maximum test accuracy achieved.
        max_val_test_f1 (float): Maximum test F1 score achieved.
        max_val_test_sens (float): Maximum test sensitivity achieved.
        max_val_test_spec (float): Maximum test specificity achieved.
        session_memory (float): Peak memory usage during the session (in MB).
        train_memory (float): Memory usage during training (in MB).
        train_time_avg (float): Average training time per epoch (in seconds).
        filename (str): The name of the file to append results to. Default is "result.csv".
        hidden_dimension (int or None): Dimension of hidden layers, if applicable.
        max_communities (int or None): Maximum number of communities, if applicable.
        remove_edges (int or None): Number of edges removed, if applicable.
    """
    filename = "results/stats/" + filename

    # Create file if it does not exist
    if not os.path.isfile(filename):
        create_empty_file_result(filename)
    
    with open(filename, 'a') as f:
        # Write results to the file
        f.write(f"{dataset},")
        f.write(f"{model_type},")
        f.write(f"{epochs},")
        f.write(f"{node_num},")
        f.write(f"{session_memory},")
        f.write(f"{train_memory},")
        f.write(f"{train_time_avg},")
        f.write(f"{max_val_acc},")
        f.write(f"{max_val_f1},")
        f.write(f"{max_val_sens},")
        f.write(f"{max_val_spec},")
        f.write(f"{max_val_test_acc},")
        f.write(f"{max_val_test_f1},")
        f.write(f"{max_val_test_sens},")
        f.write(f"{max_val_test_spec},")
        f.write(f"{hidden_dimension if hidden_dimension is not None else '-'},")
        f.write(f"{max_communities if max_communities is not None else '-'},")
        f.write(f"{remove_edges if remove_edges is not None else '-'}\n")
        