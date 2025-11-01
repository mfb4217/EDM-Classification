"""
Grid search for hyperparameter tuning
Runs multiple experiments with different configurations
"""
from config import Config
from train import Trainer
from evaluate import Evaluator
import json
import os
import pandas as pd

def create_config(base_config, **kwargs):
    """Create a new config with overridden parameters"""
    config = Config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config

def run_experiment(config, exp_id):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT {exp_id}: {config.experiment_name}")
    print('='*80)
    
    try:
        # Create experiment-specific folders
        experiment_dir = os.path.join(config.results_dir, config.experiment_name)
        config.model_dir = os.path.join(experiment_dir, "models")
        config.results_dir = os.path.join(experiment_dir, "results")
        config.logs_dir = os.path.join(experiment_dir, "logs")
        
        # Create directories
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.logs_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(experiment_dir, f"{config.experiment_name}_config.json")
        config.save_config(config_path)
        
        # Train
        trainer = Trainer(config)
        trainer.train()
        
        # Evaluate
        evaluator = Evaluator(config, trainer.model, trainer.preprocessor)
        metrics = evaluator.evaluate()
        
        # Save metrics
        metrics_path = os.path.join(experiment_dir, f"{config.experiment_name}_test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Return results
        results = {
            'experiment_id': exp_id,
            'experiment_name': config.experiment_name,
            'test_accuracy': metrics['accuracy'],
            'test_f1_normal': metrics['f1'][0],
            'test_f1_npt': metrics['f1'][1],
            'test_f1_od': metrics['f1'][2],
            'test_roc_auc': metrics.get('roc_auc', 0.0),
            'best_val_loss': float(trainer.best_val_loss),
            'best_val_acc': getattr(trainer, 'best_val_acc', 0),
        }
        
        # Add config params to results
        for key in ['max_series_length', 'include_derivatives', 'add_noise', 'noise_std',
                    'use_masking', 'mask_ratio', 'channels', 'dilations', 'strides', 
                    'kernel_size', 'dropout', 'use_depthwise_separable', 'use_residual', 
                    'activation', 'use_max_pooling', 'batch_size', 'learning_rate', 
                    'weight_decay', 'early_stopping_patience']:
            results[key] = getattr(config, key, None)
        
        return results
        
    except Exception as e:
        print(f"\nERROR in experiment {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main grid search execution"""
    
    base_config = {
        'seed': 42,
        'max_series_length': 10000,
        'normalize': True,
        'include_derivatives': True,
        'add_noise': True,
        'status_mapping': {"Normal": 0, "NPT": 1, "OD": 2},
        'num_classes': 3,
        'num_epochs': 300,
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'early_stopping_metric': "val_loss",
        'data_path': "../Data/Option 1",
        'train_path': "../Data/Option 1/Train",
        'test_path': "../Data/Option 1/Test",
        'model_dir': "models",
        'results_dir': "results",
        'logs_dir': "logs",
    }
    
    # Define grid search space
    experiments = [
        # Baseline configuration
        {
            'experiment_name': 'grid_001_baseline',
            'channels': [64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.35,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Deeper network
        {
            'experiment_name': 'grid_002_deeper',
            'channels': [32, 64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16, 32],
            'strides': [2, 2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.4,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Wider network
        {
            'experiment_name': 'grid_003_wider',
            'channels': [128, 256, 512, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.4,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Smaller, more efficient
        {
            'experiment_name': 'grid_004_compact',
            'channels': [32, 64, 128, 256],
            'dilations': [1, 2, 4, 8],
            'strides': [2, 2, 2, 2],
            'kernel_size': 5,
            'dropout': 0.3,
            'use_depthwise_separable': True,
            'use_residual': True,
            'activation': 'gelu',
            'use_max_pooling': True,
            'noise_std': 0.05,
            'batch_size': 128,
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'early_stopping_patience': 40,
        },
        
        # Larger kernel
        {
            'experiment_name': 'grid_005_large_kernel',
            'channels': [64, 128, 256, 512],
            'dilations': [1, 2, 4, 8],
            'strides': [2, 2, 2, 2],
            'kernel_size': 9,
            'dropout': 0.35,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Higher noise
        {
            'experiment_name': 'grid_006_high_noise',
            'channels': [64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.4,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.15,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Lower LR
        {
            'experiment_name': 'grid_007_low_lr',
            'channels': [64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.35,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Without derivatives
        {
            'experiment_name': 'grid_008_no_derivatives',
            'include_derivatives': False,
            'channels': [64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.35,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # ELU activation
        {
            'experiment_name': 'grid_009_elu',
            'channels': [64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.35,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'elu',
            'use_max_pooling': True,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
        
        # Without max pooling
        {
            'experiment_name': 'grid_010_avg_only',
            'channels': [64, 128, 256, 512, 512],
            'dilations': [1, 2, 4, 8, 16],
            'strides': [2, 2, 2, 2, 1],
            'kernel_size': 7,
            'dropout': 0.35,
            'use_depthwise_separable': False,
            'use_residual': True,
            'activation': 'swish',
            'use_max_pooling': False,
            'noise_std': 0.1,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'early_stopping_patience': 50,
        },
    ]
    
    print("\n" + "="*80)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    
    # Run all experiments
    results_list = []
    for exp_id, exp_config in enumerate(experiments, 1):
        config = create_config(base_config, **exp_config)
        result = run_experiment(config, exp_id)
        if result is not None:
            results_list.append(result)
    
    # Create summary dataframe
    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results = df_results.sort_values('test_accuracy', ascending=False)
        
        # Save summary
        summary_path = os.path.join("results", "grid_search_summary.csv")
        df_results.to_csv(summary_path, index=False)
        
        print("\n" + "="*80)
        print("GRID SEARCH SUMMARY")
        print("="*80)
        print(df_results.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        
        # Print top 3
        print("\n" + "="*80)
        print("TOP 3 EXPERIMENTS")
        print("="*80)
        print(df_results.head(3).to_string(index=False))
        
        # Best experiment
        best_exp = df_results.iloc[0]
        print(f"\nBEST EXPERIMENT: {best_exp['experiment_name']}")
        print(f"  Test Accuracy: {best_exp['test_accuracy']:.4f}")
        print(f"  Test F1: Normal={best_exp['test_f1_normal']:.4f}, "
              f"NPT={best_exp['test_f1_npt']:.4f}, OD={best_exp['test_f1_od']:.4f}")
        
    else:
        print("\nNo experiments completed successfully!")

if __name__ == "__main__":
    main()

