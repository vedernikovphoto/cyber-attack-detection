import os
import pandas as pd

def load_data():
    """
    Load training and testing data from CSV files.

    This function reads the training data, test data, and training labels from
    their respective CSV files and returns them as pandas DataFrames.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - train_data (DataFrame): The training data.
            - test_data (DataFrame): The test data.
            - train_labels (DataFrame): The training labels.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading the data.")

    # Define the paths to the data files
    train_data_path = os.path.join(script_dir, 'data', 'train_data.csv')
    test_data_path = os.path.join(script_dir, 'data', 'test_data.csv')
    train_labels_path = os.path.join(script_dir, 'data', 'train_labels.csv')

    # Load the data from the CSV files
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_labels = pd.read_csv(train_labels_path)

    print("Data sucesfully loaded.")
    return train_data, test_data, train_labels