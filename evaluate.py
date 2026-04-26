import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import wilcoxon
import warnings

from data_loader import load_data
from baseline import get_baseline_model
from advanced_model import get_advanced_model

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Dataset paths
DATA_DIR = "../Datasets/"
DATASETS = ["caffe.csv", "incubator-mxnet.csv", "keras.csv", "pytorch.csv", "tensorflow.csv"]

# Experiment configuration
REPEATS = 30
TEST_SIZE = 0.3

def run_experiment_on_dataset(dataset_file):
    print(f"\n--- Running Experiment on {dataset_file} ---")
    data_path = os.path.join(DATA_DIR, dataset_file)
    X, y = load_data(data_path)
    
    if len(X) == 0:
        print(f"Skipping {dataset_file} due to empty data.")
        return None
        
    baseline_metrics = {'precision': [], 'recall': [], 'f1': []}
    advanced_metrics = {'precision': [], 'recall': [], 'f1': []}
    
    for i in range(REPEATS):
        # Random 70/30 split. The loop itself provides the required 30 iterations of random splits.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=i)
        
        # Baseline Naive Bayes Model
        baseline = get_baseline_model()
        baseline.fit(X_train, y_train)
        pred_base = baseline.predict(X_test)
        
        baseline_metrics['precision'].append(precision_score(y_test, pred_base, zero_division=0))
        baseline_metrics['recall'].append(recall_score(y_test, pred_base, zero_division=0))
        baseline_metrics['f1'].append(f1_score(y_test, pred_base, zero_division=0))
        
        # Advanced SVM Model
        advanced = get_advanced_model()
        advanced.fit(X_train, y_train)
        pred_adv = advanced.predict(X_test)
        
        advanced_metrics['precision'].append(precision_score(y_test, pred_adv, zero_division=0))
        advanced_metrics['recall'].append(recall_score(y_test, pred_adv, zero_division=0))
        advanced_metrics['f1'].append(f1_score(y_test, pred_adv, zero_division=0))
        
    # Aggregate results for this dataset
    results = {
        'Dataset': dataset_file.replace('.csv', ''),
        'Baseline_Precision': np.mean(baseline_metrics['precision']),
        'Baseline_Recall': np.mean(baseline_metrics['recall']),
        'Baseline_F1': np.mean(baseline_metrics['f1']),
        'Advanced_Precision': np.mean(advanced_metrics['precision']),
        'Advanced_Recall': np.mean(advanced_metrics['recall']),
        'Advanced_F1': np.mean(advanced_metrics['f1']),
    }
    
    # Statistical test (Wilcoxon Signed-Rank Test) between the 30 runs of F1 score
    try:
        # If both models return identical arrays, wilcoxon raises an error, so we catch it
        stat, p_value = wilcoxon(baseline_metrics['f1'], advanced_metrics['f1'])
    except Exception as e:
        stat, p_value = 0, 1.0 # Not significant if identical
        
    results['P_Value_F1'] = p_value
    results['Significant'] = p_value < 0.05
    
    return results

def main():
    print("Starting Bug Report Classification Evaluation Loop...")
    final_results = []
    
    for dataset in DATASETS:
        res = run_experiment_on_dataset(dataset)
        if res:
            final_results.append(res)
            
    # Convert to DataFrame
    df = pd.DataFrame(final_results)
    print("\n\n============= FINAL EXPERIMENT RESULTS =============")
    print(df.to_string(index=False))
    
    # Save to disk
    df.to_csv("evaluation_results.csv", index=False)
    print("\nResults successfully saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
