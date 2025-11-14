"""
Complete Pipeline
Orchestrates all steps: augmentation -> training -> threshold optimization -> evaluation -> runtime
"""
import json
import os
import time
import numpy as np

from augmentation import generate_augmented_data
from training import create_validation_split, train_ensemble
from thresholds import optimize_thresholds, save_thresholds
from evaluation import evaluate_test_set, print_individual_model_metrics
from runtime import evaluate_runtime


def convert_numpy_types(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with all NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def run_complete_pipeline(config_path):
    """
    Run complete pipeline from data augmentation to final evaluation.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Dictionary with all results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    print("="*80)
    print("FINAL STATUS CLASSIFICATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"Experiment name: {config_dict['experiment_name']}")
    
    start_time = time.time()
    
    # Step 1: Data augmentation
    generate_augmented_data(config_dict)
    
    # Step 2: Create validation split
    predefined_split = create_validation_split(config_dict)
    
    # Step 3: Train ensemble
    all_results, ensemble_probs, labels, class_names = train_ensemble(config_dict, predefined_split)
    
    # Step 4: Optimize thresholds (only NPT, argmax for Normal and OD)
    threshold_config = config_dict['threshold_optimization']
    optimize_only_npt = threshold_config.get('optimize_only_npt', True)  # Default: True
    threshold_search = optimize_thresholds(
        ensemble_probs['validation'],
        labels['validation'],
        class_names,
        fpr_limits=threshold_config['fpr_limits'],
        fnr_limits=threshold_config['fnr_limits'],
        min_accuracy=threshold_config['min_accuracy'],
        grid_start=threshold_config['threshold_grid_start'],
        grid_end=threshold_config['threshold_grid_end'],
        grid_steps=threshold_config['threshold_grid_steps'],
        optimize_only_npt=optimize_only_npt
    )
    
    thresholds = threshold_search['thresholds']
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    if optimize_only_npt:
        print("\nOptimizing only NPT threshold (argmax for Normal and OD):")
    else:
        print("\nOptimized thresholds per class:")
    for idx, class_name in enumerate(class_names):
        if optimize_only_npt and class_name != 'NPT':
            print(f"  {class_name}: argmax (no threshold)")
        else:
            print(f"  {class_name}: {thresholds[idx]:.3f}")
    
    if threshold_search.get('constraints_met', False):
        print("\nValidation metrics at selected thresholds:")
        print(f"  Accuracy: {threshold_search['accuracy']:.4f}")
        for idx, class_name in enumerate(class_names):
            print(f"  {class_name}: FPR={threshold_search['fpr'][idx]:.4f}, FNR={threshold_search['fnr'][idx]:.4f}")
        print("  Constraint status: met")
    else:
        print("\nWarning: Constraints not fully met, using fallback thresholds")
    
    # Save thresholds
    results_dir = config_dict['output_paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    thresholds_path = os.path.join(results_dir, f"{config_dict['experiment_name']}_thresholds.json")
    save_thresholds(thresholds, class_names, thresholds_path)
    
    # Step 5: Evaluate test set
    test_metrics = evaluate_test_set(ensemble_probs, labels, thresholds, class_names, config_dict)
    print_individual_model_metrics(all_results)
    
    # Step 6: Evaluate runtime (optional)
    runtime_metrics = None
    try:
        runtime_metrics = evaluate_runtime(config_dict, all_results, thresholds)
    except Exception as e:
        print(f"\nWarning: Runtime evaluation failed: {e}")
    
    total_time = time.time() - start_time
    
    # Save final results
    final_results = {
        'experiment_name': config_dict['experiment_name'],
        'configuration': config_dict,
        'thresholds': thresholds.tolist(),
        'thresholds_path': thresholds_path,
        'threshold_search': {
            'constraints_met': bool(threshold_search.get('constraints_met', False)),  # Convert to Python bool
            'validation_accuracy': float(threshold_search.get('accuracy', 0)) if threshold_search.get('accuracy') is not None else None,
            'validation_fpr': threshold_search.get('fpr', []).tolist() if threshold_search.get('fpr') is not None else None,
            'validation_fnr': threshold_search.get('fnr', []).tolist() if threshold_search.get('fnr') is not None else None
        },
        'test_metrics': test_metrics,
        'runtime_metrics': runtime_metrics,
        'individual_models': [
            {
                'run_id': idx + 1,
                'test_accuracy': float(r['accuracy']),
                'test_roc_auc': float(r['roc_auc'])
            }
            for idx, r in enumerate(all_results)
        ],
        'pipeline_total_time_seconds': float(total_time)
    }
    
    # Convert all NumPy types to native Python types for JSON serialization
    final_results = convert_numpy_types(final_results)
    
    results_path = os.path.join(results_dir, f"{config_dict['experiment_name']}_final_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Total pipeline time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {results_path}")
    print(f"Thresholds saved to: {thresholds_path}")
    
    return final_results

