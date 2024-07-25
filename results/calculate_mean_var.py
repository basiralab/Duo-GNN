"""
Created on 23 July 2024

@author: kevin
"""

import pandas as pd
import sys

def calculate_mean_and_variance(filename):
    """
    Calculates mean and variance of specified columns grouped by unique running settings.

    Args:
        filename (str): Path to the CSV file containing the data.
    """
    # Read the data from a CSV file
    data = pd.read_csv(filename)

    # Define the columns that represent the unique running setting
    grouping_columns = ['Dataset', 'Model_Type', 'Node_Num', 'Hidden_Dimension', 'Max_Communities', 'Remove_Edges', 'Epochs','Topological_Measure','Make_Unbalanced','Dense']

    # Define the columns for which we want to calculate the mean and variance
    measurement_columns = ['Session_Memory', 'Train_Memory', 'Train_Time_Avg', 'Max_Val_Acc', 'Max_Val_F1', 
                           'Max_Val_Sens', 'Max_Val_Spec', 'Max_Val_Test_Acc', 'Max_Val_Test_F1', 
                           'Max_Val_Test_Sens', 'Max_Val_Test_Spec']

    # Group the data by the unique running setting
    grouped_data = data.groupby(grouping_columns)

    # Calculate the mean and variance for each group
    mean_data = grouped_data[measurement_columns].mean().reset_index()
    variance_data = grouped_data[measurement_columns].std().reset_index()

    # Calculate the number of rows in each group
    count_data = grouped_data.size().reset_index(name='Count')

    # Merge the mean, variance, and count data
    mean_data = mean_data.merge(count_data, on=grouping_columns)
    variance_data = variance_data.merge(count_data, on=grouping_columns)

    # Print the results
    print("Mean values for each group with counts:\n", mean_data)
    print("\nVariance values for each group with counts:\n", variance_data)

if __name__ == "__main__":
    # Check if the filename is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    else:
        filename = sys.argv[1]
        calculate_mean_and_variance(filename)
