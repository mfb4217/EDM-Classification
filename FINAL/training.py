"""
Training Module
Train ensemble of models for status classification
"""
import os
import numpy as np
import joblib
from config import dict_to_config
from preprocessing import DataPreprocessor
from utils import create_config
from inference import save_preprocessor

# We need to wrap run_single_model to also return preprocessor
def run_single_model_with_preprocessor(config, exp_id):
    """Wrapper around run_single_model that also returns preprocessor"""
    from train import Trainer
    
    # Replicate the logic from run_ensemble.py but capture preprocessor
    from tqdm import tqdm
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    try:
        # Simplified structure: save model directly in experiment_dir (no nested models/ folder)
        experiment_dir = os.path.join(config.results_dir, config.experiment_name)
        config.model_dir = experiment_dir  # Save directly in model directory
        config.results_dir = experiment_dir
        config.logs_dir = None  # Not used - don't create logs directory
        
        # Verify paths are set
        if config.train_path is None:
            raise ValueError(f"config.train_path is None for experiment {config.experiment_name}")
        
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Don't save config.json per model (redundant - final_results.json has full config)
        
        trainer = Trainer(config)
        trainer.train()
        
        train_probabilities, train_labels = trainer.get_train_predictions()
        val_probabilities, val_labels = trainer.get_validation_predictions()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preprocessor = trainer.preprocessor
        X_test, y_test = preprocessor.preprocess_test(config.test_path, variable_length=False)
        
        all_probabilities = []
        all_predictions = []
        all_ground_truth = []
        
        trainer.model.eval()
        with torch.no_grad():
            for x, y in tqdm(zip(X_test, y_test), total=len(X_test)):
                x_tensor = torch.FloatTensor(x).transpose(0, 1).unsqueeze(0).to(device)
                logits = trainer.model(x_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                
                all_probabilities.append(probs.cpu().numpy()[0])
                all_predictions.append(pred.cpu().numpy()[0])
                all_ground_truth.append(y)
        
        y_prob = np.array(all_probabilities)
        y_pred = np.array(all_predictions)
        y_true = np.array(all_ground_truth)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.0
        
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
            'train_probabilities': train_probabilities,
            'y_train': train_labels,
            'val_probabilities': val_probabilities,
            'y_val': val_labels,
            'class_names': class_names,
            'model': trainer.model,
            'preprocessor': preprocessor  # IMPORTANT: return preprocessor
        }
        
    except Exception as e:
        print(f"\nERROR in model {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_validation_split(config_dict):
    """
    Create train/validation split ensuring validation is not in Option 2.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        predefined_split: Dictionary with 'train' and 'val' file lists
    """
    print("\n" + "="*80)
    print("CREATE TRAIN/VALIDATION SPLIT")
    print("="*80)
    
    # Create config without augmented data for split creation
    base_config_dict = config_dict.copy()
    base_config_dict['data_paths'] = config_dict['data_paths'].copy()
    base_config_dict['data_paths']['augmented_data_path'] = None
    base_config_dict['experiment_name'] = 'ensemble_split'
    
    split_config = dict_to_config(base_config_dict)
    split_preprocessor = DataPreprocessor(split_config)
    split_preprocessor.preprocess_train(split_config.train_path)
    
    predefined_split = getattr(split_config, 'predefined_split', None)
    if predefined_split is None:
        raise RuntimeError("Failed to generate predefined train/validation split.")
    
    print(f"Split created: {len(predefined_split['train'])} train, {len(predefined_split['val'])} validation")
    
    return predefined_split


def train_ensemble(config_dict, predefined_split):
    """
    Train ensemble of models.
    
    Args:
        config_dict: Configuration dictionary
        predefined_split: Train/validation split dictionary
        
    Returns:
        all_results: List of results from each model
        ensemble_probs: Dictionary with averaged probabilities for train/val/test
        labels: Dictionary with labels for train/val/test
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("TRAIN ENSEMBLE")
    print("="*80)
    
    ensemble_config = config_dict['ensemble']
    num_models = ensemble_config['num_models']
    model_seeds = ensemble_config['model_seeds']
    
    # Create base config
    base_config = dict_to_config(config_dict)
    base_config.predefined_split = predefined_split
    
    all_results = []
    all_probabilities = []
    y_test = None
    class_names = None
    preprocessor_saved = False
    
    for run_id in range(1, num_models + 1):
        # Create a new config object with updated experiment name and seed
        config = dict_to_config(config_dict)
        config.experiment_name = f"{config_dict['experiment_name']}_model_{run_id:02d}"
        config.seed = model_seeds[run_id - 1] if run_id <= len(model_seeds) else config_dict['seed'] + run_id
        config.predefined_split = predefined_split  # Ensure split is preserved
        
        result = run_single_model_with_preprocessor(config, run_id)
        
        if result is not None:
            all_results.append(result)
            all_probabilities.append(result['probabilities'])
            if y_test is None:
                y_test = result['y_test']
                class_names = result['class_names']
            
            # Save preprocessor/scaler after first model (all models use same scaler)
            # CRITICAL: Prevents data leakage - scaler must be same as training
            if not preprocessor_saved and 'preprocessor' in result:
                results_dir = config_dict['output_paths']['results_dir']
                experiment_name = config_dict['experiment_name']
                os.makedirs(results_dir, exist_ok=True)
                scaler_path = os.path.join(results_dir, f"{experiment_name}_scaler.pkl")
                save_preprocessor(result['preprocessor'], scaler_path)
                preprocessor_saved = True
                print(f"\nScaler saved to: {scaler_path} (prevents data leakage in inference)")
    
    if not all_results:
        raise RuntimeError("No models trained successfully!")
    
    print(f"\nEnsemble created: {len(all_results)} models")
    print(f"Test data: {len(y_test)} samples")
    
    # Aggregate probabilities
    train_probabilities = np.array([r['train_probabilities'] for r in all_results])
    val_probabilities = np.array([r['val_probabilities'] for r in all_results])
    test_probabilities = np.array(all_probabilities)
    
    ensemble_probs = {
        'train': np.mean(train_probabilities, axis=0),
        'validation': np.mean(val_probabilities, axis=0),
        'test': np.mean(test_probabilities, axis=0)
    }
    
    labels = {
        'train': all_results[0]['y_train'],
        'validation': all_results[0]['y_val'],
        'test': y_test
    }
    
    return all_results, ensemble_probs, labels, class_names



