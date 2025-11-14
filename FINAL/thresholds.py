"""
Threshold Optimization Module
Optimize per-class probability thresholds on validation set
"""
import sys
import os
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'Status Classification'))

from run_ensemble import apply_thresholds, compute_class_rates, check_constraints

# Default limits
FPR_LIMITS = {'Normal': 0.15, 'NPT': 0.05, 'OD': 0.05}
FNR_LIMITS = {'Normal': 0.10, 'NPT': 0.30, 'OD': 0.20}


def apply_thresholds_selective(probabilities, thresholds, class_names, optimize_only_npt=True):
    """
    Apply thresholds selectively: argmax for Normal and OD, threshold only for NPT.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) with class probabilities
        thresholds: Array of thresholds (only NPT threshold is used if optimize_only_npt=True)
        class_names: List of class names
        optimize_only_npt: If True, only use threshold for NPT, use argmax for others
        
    Returns:
        Array of predictions
    """
    if optimize_only_npt:
        # Find NPT index
        npt_idx = class_names.index('NPT') if 'NPT' in class_names else 1
        
        predictions = []
        for probs in probabilities:
            # First, do argmax excluding NPT (for Normal and OD)
            other_indices = [i for i in range(len(class_names)) if i != npt_idx]
            other_probs = probs[other_indices]
            argmax_other = other_indices[np.argmax(other_probs)]
            argmax_prob = probs[argmax_other]
            
            # Check if NPT meets its threshold
            npt_prob = probs[npt_idx]
            npt_threshold = thresholds[npt_idx]
            
            # Use NPT only if it meets threshold AND has higher probability than argmax
            if npt_prob >= npt_threshold and npt_prob > argmax_prob:
                predictions.append(npt_idx)
            else:
                predictions.append(argmax_other)
        
        return np.array(predictions, dtype=int)
    else:
        # Use original apply_thresholds for all classes
        return apply_thresholds(probabilities, thresholds)


def optimize_thresholds(probabilities, y_true, class_names, 
                      fpr_limits=None, fnr_limits=None, 
                      min_accuracy=0.90, grid_start=0.1, grid_end=0.95, grid_steps=36,
                      optimize_only_npt=True):
    """
    Find optimal thresholds. If optimize_only_npt=True, only optimizes NPT threshold,
    using argmax for Normal and OD.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) with class probabilities
        y_true: Array of true class labels
        class_names: List of class names
        fpr_limits: Dictionary with FPR limits per class (default: FPR_LIMITS)
        fnr_limits: Dictionary with FNR limits per class (default: FNR_LIMITS)
        min_accuracy: Minimum accuracy threshold
        grid_start: Start of threshold grid
        grid_end: End of threshold grid
        grid_steps: Number of steps in threshold grid
        optimize_only_npt: If True, only optimize NPT threshold (default: True)
        
    Returns:
        Dictionary with:
            - thresholds: Optimized thresholds array (only NPT non-zero if optimize_only_npt=True)
            - accuracy: Validation accuracy at these thresholds
            - fpr: FPR per class
            - fnr: FNR per class
            - constraints_met: Whether constraints are satisfied
    """
    if fpr_limits is None:
        fpr_limits = FPR_LIMITS
    if fnr_limits is None:
        fnr_limits = FNR_LIMITS
    
    if optimize_only_npt:
        # Only optimize NPT threshold
        npt_idx = class_names.index('NPT') if 'NPT' in class_names else 1
        threshold_grid = np.linspace(grid_start, grid_end, grid_steps)
        
        best = None
        
        for npt_threshold in threshold_grid:
            # Create thresholds array: 0.0 for Normal and OD, npt_threshold for NPT
            thresholds = np.zeros(len(class_names), dtype=np.float32)
            thresholds[npt_idx] = npt_threshold
            
            y_pred = apply_thresholds_selective(probabilities, thresholds, class_names, optimize_only_npt=True)
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy < min_accuracy:
                continue
            
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            fpr, fnr = compute_class_rates(cm)
            
            if not check_constraints(fpr, fnr, class_names):
                continue
            
            if best is None or accuracy > best['accuracy']:
                best = {
                    'thresholds': thresholds.copy(),
                    'accuracy': accuracy,
                    'fpr': fpr,
                    'fnr': fnr,
                    'active_thresholds': 1  # Only NPT
                }
            elif accuracy == best['accuracy']:
                current_penalty = (fpr + fnr).sum()
                best_penalty = (best['fpr'] + best['fnr']).sum()
                if current_penalty < best_penalty:
                    best = {
                        'thresholds': thresholds.copy(),
                        'accuracy': accuracy,
                        'fpr': fpr,
                        'fnr': fnr,
                        'active_thresholds': 1
                    }
        
        if best is None:
            print("Warning: Could not find NPT threshold meeting the specified constraints. Falling back to argmax for all classes.")
            return {
                'thresholds': np.zeros(len(class_names), dtype=np.float32),
                'accuracy': None,
                'fpr': None,
                'fnr': None,
                'constraints_met': False
            }
        
        best['constraints_met'] = True
        return best
    
    else:
        # Original behavior: optimize all thresholds
        threshold_grid = np.linspace(grid_start, grid_end, grid_steps)
        best = None
        
        for thresholds in product(threshold_grid, repeat=len(class_names)):
            thresholds = np.array(thresholds, dtype=np.float32)
            y_pred = apply_thresholds(probabilities, thresholds)
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy < min_accuracy:
                continue
            
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            fpr, fnr = compute_class_rates(cm)
            
            if not check_constraints(fpr, fnr, class_names):
                continue
            
            thresholds_non_default = np.sum(thresholds > 0.0)
            if best is None or accuracy > best['accuracy']:
                best = {
                    'thresholds': thresholds,
                    'accuracy': accuracy,
                    'fpr': fpr,
                    'fnr': fnr,
                    'active_thresholds': thresholds_non_default
                }
            elif accuracy == best['accuracy']:
                if thresholds_non_default < best['active_thresholds']:
                    best = {
                        'thresholds': thresholds,
                        'accuracy': accuracy,
                        'fpr': fpr,
                        'fnr': fnr,
                        'active_thresholds': thresholds_non_default
                    }
                else:
                    current_penalty = (fpr + fnr).sum()
                    best_penalty = (best['fpr'] + best['fnr']).sum()
                    if thresholds_non_default == best['active_thresholds'] and current_penalty < best_penalty:
                        best = {
                            'thresholds': thresholds,
                            'accuracy': accuracy,
                            'fpr': fpr,
                            'fnr': fnr,
                            'active_thresholds': thresholds_non_default
                        }
        
        if best is None:
            print("Warning: Could not find thresholds meeting the specified constraints. Falling back to default argmax behavior.")
            return {
                'thresholds': np.zeros(len(class_names), dtype=np.float32),
                'accuracy': None,
                'fpr': None,
                'fnr': None,
                'constraints_met': False
            }
        
        best['constraints_met'] = True
        return best


def save_thresholds(thresholds, class_names, output_path):
    """
    Save optimized thresholds to JSON file.
    
    Args:
        thresholds: Array of thresholds
        class_names: List of class names
        output_path: Path to save JSON file
    """
    import json
    
    thresholds_dict = {name: float(thresholds[i]) for i, name in enumerate(class_names)}
    
    with open(output_path, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    print(f"Thresholds saved to: {output_path}")


def load_thresholds(input_path, class_names):
    """
    Load thresholds from JSON file.
    
    Args:
        input_path: Path to JSON file
        class_names: List of class names in order
        
    Returns:
        Array of thresholds
    """
    import json
    
    with open(input_path, 'r') as f:
        thresholds_dict = json.load(f)
    
    thresholds = np.array([thresholds_dict[name] for name in class_names], dtype=np.float32)
    return thresholds

