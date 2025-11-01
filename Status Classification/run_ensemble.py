"""
Train ensemble of models and evaluate
Run the best configuration multiple times with different seeds
Create ensemble by averaging predictions
"""
from config import Config
from train import Trainer
from evaluate import Evaluator
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
from preprocessing import DataPreprocessor

def create_config(base_config, **kwargs):
    """Create a new config with overridden parameters"""
    config = Config()
    # First set base_config values
    for key, value in base_config.items():
        setattr(config, key, value)
    # Then override with kwargs
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config

def run_single_model(config, exp_id):
    """Train and evaluate a single model, return metrics and probabilities"""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL {exp_id}")
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
        
        # Evaluate and collect probabilities
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preprocessor = trainer.preprocessor
        X_test, y_test = preprocessor.preprocess_test(config.test_path, variable_length=False)
        
        all_probabilities = []
        all_predictions = []
        all_ground_truth = []
        
        trainer.model.eval()
        with torch.no_grad():
            for x, y in tqdm(zip(X_test, y_test), total=len(X_test)):
                # Convert to tensor
                x_tensor = torch.FloatTensor(x).transpose(0, 1).unsqueeze(0).to(device)
                
                # Predict
                logits = trainer.model(x_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                
                all_probabilities.append(probs.cpu().numpy()[0])
                all_predictions.append(pred.cpu().numpy()[0])
                all_ground_truth.append(y)
        
        # Convert to numpy
        y_prob = np.array(all_probabilities)
        y_pred = np.array(all_predictions)
        y_true = np.array(all_ground_truth)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.0
        
        # Get class names
        class_names = config.get_class_names()
        
        print(f"\nModel {exp_id} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  F1: Normal={f1[0]:.4f}, NPT={f1[1]:.4f}, OD={f1[2]:.4f}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_normal': f1[0],
            'f1_npt': f1[1],
            'f1_od': f1[2],
            'probabilities': y_prob,
            'y_test': y_true,
            'class_names': class_names,
            'model': trainer.model
        }
        
    except Exception as e:
        print(f"\nERROR in model {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_ensemble(all_probabilities, y_true, class_names):
    """Evaluate ensemble by averaging probabilities"""
    # Average probabilities
    avg_probabilities = np.mean(all_probabilities, axis=0)
    ensemble_predictions = np.argmax(avg_probabilities, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_true, ensemble_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, ensemble_predictions, average=None, zero_division=0)
    cm = confusion_matrix(y_true, ensemble_predictions)
    
    try:
        roc_auc = roc_auc_score(y_true, avg_probabilities, multi_class='ovr', average='macro')
    except:
        roc_auc = 0.0
    
    print(f"\n{'='*80}")
    print("ENSEMBLE RESULTS")
    print('='*80)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print(f"\nPer-class metrics:")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:<10} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10.0f}")
    
    print(f"\nConfusion Matrix:")
    print(f"{'':>10}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, ensemble_predictions, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'confusion_matrix': cm.tolist()
    }

def main():
    """Main ensemble execution"""
    
    # Best configuration from grid search (grid_008_no_derivatives)
    base_config = {
        'seed': 42,
        'max_series_length': 10000,
        'normalize': True,
        'include_derivatives': False,  # NO DERIVATIVES
        'status_mapping': {"Normal": 0, "NPT": 1, "OD": 2},
        'num_classes': 3,
        'use_fixed_class_weights': True,  # Use fixed weights to avoid imbalance from synthetic data
        'fixed_class_distribution': {"Normal": 220, "NPT": 66, "OD": 75},  # Expected distribution
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
        'channels': [64, 128, 256, 512, 512],
        'dilations': [1, 2, 4, 8, 16],
        'strides': [2, 2, 2, 2, 1],
        'kernel_size': 7,
        'dropout': 0.3,
        'use_depthwise_separable': False,
        'use_residual': True,
        'activation': 'swish',
        'use_max_pooling': True,
        'add_noise': True,
        'noise_std': 0.08,
        'use_masking': False,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'weight_decay': 0.004,
        'early_stopping_patience': 50,
        'dense_hidden_ratio': 1.0,
        'dense_hidden_min': 256,
    }
    
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING (10 MODELS)")
    print("="*80)
    print("Configuration: grid_008_no_derivatives (Best from grid search)")
    print("  - include_derivatives: False")
    print("  - channels: [64, 128, 256, 512, 512]")
    print("  - dropout: 0.35")
    print("  - noise_std: 0.1")
    print("  - activation: swish")
    
    # Train 10 models with different seeds
    all_probabilities = []
    all_results = []
    y_test = None
    class_names = None
    
    for run_id in range(1, 11):
        config = create_config(base_config, 
                              experiment_name=f'ensemble_run_{run_id:02d}',
                              seed=42 + run_id)  # Different seed for each run
        
        result = run_single_model(config, run_id)
        
        if result is not None:
            all_results.append(result)
            all_probabilities.append(result['probabilities'])
            # Save y_test and class_names from first result
            if y_test is None:
                y_test = result['y_test']
                class_names = result['class_names']
    
    if not all_results:
        print("\nNo models trained successfully!")
        return
    
    print(f"\nTest data: {len(y_test)} samples")
    
    # Convert all_probabilities to numpy array
    all_probabilities = np.array(all_probabilities)  # (n_models, n_samples, n_classes)
    
    # Individual model statistics
    print(f"\n{'='*80}")
    print("INDIVIDUAL MODELS SUMMARY")
    print('='*80)
    individual_df = pd.DataFrame(all_results)
    print(f"\n{'Run':<5} {'Accuracy':>10} {'ROC AUC':>10} {'F1 Normal':>12} {'F1 NPT':>10} {'F1 OD':>10}")
    print("-" * 75)
    for i, result in enumerate(all_results, 1):
        print(f"{i:<5} {result['accuracy']:>10.4f} {result['roc_auc']:>10.4f} "
              f"{result['f1_normal']:>12.4f} {result['f1_npt']:>10.4f} {result['f1_od']:>10.4f}")
    
    print(f"\nIndividual Models Statistics:")
    print(f"  Mean Accuracy: {individual_df['accuracy'].mean():.4f} ± {individual_df['accuracy'].std():.4f}")
    print(f"  Mean ROC AUC: {individual_df['roc_auc'].mean():.4f} ± {individual_df['roc_auc'].std():.4f}")
    print(f"  Mean F1 Normal: {individual_df['f1_normal'].mean():.4f} ± {individual_df['f1_normal'].std():.4f}")
    print(f"  Mean F1 NPT: {individual_df['f1_npt'].mean():.4f} ± {individual_df['f1_npt'].std():.4f}")
    print(f"  Mean F1 OD: {individual_df['f1_od'].mean():.4f} ± {individual_df['f1_od'].std():.4f}")
    
    # Save individual results
    individual_path = os.path.join("results", "ensemble_individual_results.csv")
    individual_df.to_csv(individual_path, index=False)
    print(f"\nIndividual results saved to: {individual_path}")
    
    # Ensemble evaluation
    ensemble_metrics = evaluate_ensemble(all_probabilities, y_test, class_names)
    
    # Save ensemble results
    ensemble_path = os.path.join("results", "ensemble_results.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    print(f"\nEnsemble results saved to: {ensemble_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("ENSEMBLE VS INDIVIDUAL COMPARISON")
    print('='*80)
    print(f"Individual Models (mean):")
    print(f"  Accuracy: {individual_df['accuracy'].mean():.4f} ± {individual_df['accuracy'].std():.4f}")
    print(f"  ROC AUC: {individual_df['roc_auc'].mean():.4f} ± {individual_df['roc_auc'].std():.4f}")
    print(f"\nEnsemble:")
    print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  ROC AUC: {ensemble_metrics['roc_auc']:.4f}")
    print(f"  Improvement: {ensemble_metrics['accuracy'] - individual_df['accuracy'].mean():.4f}")

if __name__ == "__main__":
    main()

