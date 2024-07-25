"""
Created on 23 July 2024

@author: Kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_deltas(deltas, num_nodes_to_plot=5):
    """
    Plot the feature variation (delta) over aggregation steps for a specified number of nodes.

    Parameters:
    - deltas (list of np.array): List where each element is an array of deltas for a specific node.
    - num_nodes_to_plot (int): Number of nodes to plot. Defaults to 5.

    This function creates a line plot where each line represents the feature variation over aggregation steps for each node.
    """
    plt.figure(figsize=(10, 6))
    for i in range(num_nodes_to_plot):
        plt.plot(deltas[i], label=f'Node {i}')
    plt.xlabel('Aggregation Step')
    plt.ylabel('Feature Variation (Delta)')
    plt.title('Feature Variation Over Aggregation Steps')
    plt.legend()
    plt.show()

def plot_mse_results(mse_results, filename='mse_results.png'):
    """
    Plot the Mean Squared Error (MSE) results for each node and save the plot as an image.

    Parameters:
    - mse_results (list of float): List of MSE values for each node.
    - filename (str): Name of the file where the plot will be saved. Defaults to 'mse_results.png'.

    This function creates a bar plot where each bar represents the MSE for a specific node and saves the plot to a file.
    """
    num_nodes = len(mse_results)
    nodes = np.arange(num_nodes)
    filename = "results/images/" + filename

    plt.figure(figsize=(10, 6))
    plt.bar(nodes, mse_results, color='skyblue')
    plt.xlabel('Nodes')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Mean Squared Error for Each Node')
    plt.xticks(nodes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(filename)
    plt.show()

def calculate_homophily_ratios(graph):
    """
    Calculate the homophily ratios for all edges in the graph.

    Parameters:
    - graph (networkx.Graph): The input graph where each node has 'x' (features) and 'y' (label) attributes.

    Returns:
    - homophily_ratios (list of float): List of homophily ratios for each edge in the graph.

    Homophily ratio is calculated as a combination of feature similarity and label similarity between connected nodes.
    """
    homophily_ratios = []

    for edge in graph.edges():
        node1, node2 = edge
        # Extract features and labels for both nodes
        x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
        x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
        
        # Calculate similarity in features (using cosine similarity)
        feature_similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        
        # Calculate similarity in labels
        label_similarity = 1 if y1 == y2 else 0
        
        # Compute homophily ratio as a weighted combination of feature and label similarity
        homophily_ratio = 0.5 * feature_similarity + 0.5 * label_similarity
        
        homophily_ratios.append(homophily_ratio)
    
    return homophily_ratios

def plot_homophily_distributions(graphs, stages, dataset):
    """
    Plot the distribution of homophily ratios across different stages.

    Parameters:
    - graphs (list of networkx.Graph): List of three graphs corresponding to different stages.
    - stages (list of str): List of stage names for labeling the plot.
    - dataset (str): Name of the dataset for saving the plot file.

    This function calculates homophily ratios for each graph and plots their distributions using KDE plots.
    The plot is saved to a file with a name based on the dataset.
    """
    if len(graphs) != 3 or len(stages) != 3:
        raise ValueError("You must provide exactly three graphs and three stages.")
    
    homophily_data = [calculate_homophily_ratios(graph) for graph in graphs]
    colors = ['#2E8B57', '#FF6347', '#8670FD']  # Different colors for better visual distinction

    plt.figure(figsize=(12, 6))

    for i in range(3):
        sns.kdeplot(homophily_data[i], color=colors[i], fill=True, label=f'{stages[i]}', common_norm=True)

    plt.xlabel('Homophily Ratio')
    plt.ylabel('Density')
    plt.title('Homophily Ratios Distribution Across Stages')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the plot to a file
    output_file = f'results/images/distribution_homophily_{dataset}.png'
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_degree_distribution(G_old, G_new, output_file):
    """
    Plot and compare the degree distributions of two graphs.

    Parameters:
    - G_old (networkx.Graph): The first graph for comparison.
    - G_new (networkx.Graph): The second graph for comparison.
    - output_file (str): Name of the file where the plot will be saved.

    This function plots the degree distributions of the two graphs on the same plot for comparison and saves the plot to a file.
    """
    output_file = 'results/images/' + output_file

    # Get degree sequences for both graphs and sort them
    degree_sequence_old = sorted((d for n, d in G_old.degree()), reverse=True)
    degree_sequence_new = sorted((d for n, d in G_new.degree()), reverse=True)

    plt.figure(figsize=(10, 6))

    # Plot the degree distribution for the old graph
    plt.plot(degree_sequence_old, 'o', label='G_old', alpha=0.5)
    
    # Plot the degree distribution for the new graph
    plt.plot(degree_sequence_new, 'o', label='G_new', alpha=0.5)

    plt.title("Degree Distribution Comparison")
    plt.xlabel("Rank Node")
    plt.ylabel("Degree")
    plt.yscale('log')
    plt.legend()

    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()
