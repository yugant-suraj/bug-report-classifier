import pandas as pd
import numpy as np
import os

def load_data(csv_path):
    """
    Loads bug reports from a given CSV file path.
    Extracts the textual features (Title, Body, Comments) and Class.
    Returns:
        X (list of str): The combined text for each report.
        y (np.array): The label (0 or 1) for each report.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return [], np.array([])
    
    # Fill missing text fields with empty strings
    df['Title'] = df['Title'].fillna('')
    df['Body'] = df['Body'].fillna('')
    df['Comments'] = df['Comments'].fillna('')
    
    # Combine title, body and comments to form a comprehensive text representation
    # This maximizes the information our ML model can learn from
    X = (df['Title'] + " " + df['Body'] + " " + df['Comments']).tolist()
    
    # Target variable
    y = df['class'].values
    
    return X, y
