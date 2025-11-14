"""
Evaluation Module
Evaluate ensemble on test set with optimized thresholds
"""
import numpy as np
from utils import evaluate_with_thresholds, check_constraints, compute_class_rates
from thresholds import apply_thresholds_selective
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def evaluate_test_set(ensemble_probs, labels, thresholds, class_names, config_dict):
    """
    Evaluate ensemble on test set with optimized thresholds.
    Uses selective threshold application (argmax for Normal/OD, threshold for NPT).
    
    Args:
        ensemble_probs: Dictionary with probabilities for train/val/test
        labels: Dictionary with labels for train/val/test
        thresholds: Array of optimized thresholds
        class_names: List of class names
        config_dict: Configuration dictionary (for constraint limits)
        
    Returns:
        Dictionary with test metrics
    """
    print("\n" + "="*80)
    print("EVALUATE ON TEST SET")
    print("="*80)
    
    # Use selective threshold application (argmax for Normal/OD, threshold for NPT)
    y_pred = apply_thresholds_selective(ensemble_probs['test'], thresholds, class_names, optimize_only_npt=True)
    y_true = labels['test']
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fpr, fnr = compute_class_rates(cm)
    
    try:
        roc_auc = roc_auc_score(y_true, ensemble_probs['test'], multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0
    
    test_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fpr': fpr.tolist(),
        'fnr': fnr.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\nTest Set Results (with thresholds):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"\n  FPR (per class):")
    for name, value in zip(class_names, test_metrics['fpr']):
        print(f"    {name}: {value:.4f}")
    print(f"\n  FNR (per class):")
    for name, value in zip(class_names, test_metrics['fnr']):
        print(f"    {name}: {value:.4f}")
    
    # Check constraints
    fpr_limits = config_dict['threshold_optimization']['fpr_limits']
    fnr_limits = config_dict['threshold_optimization']['fnr_limits']
    constraints_met = check_constraints(
        np.array(test_metrics['fpr']),
        np.array(test_metrics['fnr']),
        class_names
    )
    status = "met" if constraints_met else "NOT met"
    print(f"\n  Constraint status (Test): {status}")
    
    return test_metrics


def print_individual_model_metrics(all_results):
    """
    Print individual model metrics.
    
    Args:
        all_results: List of results from each model
    """
    print(f"\n  Individual Models Test ROC AUC:")
    individual_roc_aucs = []
    for idx, result in enumerate(all_results, 1):
        individual_roc_aucs.append(result['roc_auc'])
        print(f"    Model {idx}: {result['roc_auc']:.4f}")
    
    if individual_roc_aucs:
        mean_roc = np.mean(individual_roc_aucs)
        std_roc = np.std(individual_roc_aucs)
        print(f"\n  Individual mean ± std: {mean_roc:.4f} ± {std_roc:.4f}")

