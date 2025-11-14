"""
Training Module
Train ensemble of models for status classification
"""
import sys
import os
import numpy as np
import joblib

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'Status Classification'))

from config import Config
from preprocessing import DataPreprocessor
from run_ensemble import create_config
from inference import save_preprocessor

# We need to wrap run_single_model to also return preprocessor
def run_single_model_with_preprocessor(config, exp_id):
    """Wrapper around run_single_model that also returns preprocessor"""
    from run_ensemble import run_single_model as original_run_single_model
    from train import Trainer
    
    # Replicate the logic from run_ensemble.py but capture preprocessor
    import os
    from tqdm import tqdm
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    try:
        # Simplified structure: save model directly in experiment_dir (no nested models/ folder)
        experiment_dir = os.path.join(config.results_dir, config.experiment_name)
        config.model_dir = experiment_dir  # Save directly in model directory
        config.results_dir = experiment_dir
        config.logs_dir = None  # Not used - don't create logs directory
        
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
    base_config = _create_config_from_dict(config_dict).__dict__
    base_config['augmented_data_path'] = None
    base_config['experiment_name'] = 'ensemble_split'
    
    split_config = create_config(base_config)
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
    base_config = _create_config_from_dict(config_dict).__dict__
    base_config['predefined_split'] = predefined_split
    
    all_results = []
    all_probabilities = []
    y_test = None
    class_names = None
    preprocessor_saved = False
    
    for run_id in range(1, num_models + 1):
        config = create_config(
            base_config,
            experiment_name=f"{config_dict['experiment_name']}_model_{run_id:02d}",
            seed=model_seeds[run_id - 1] if run_id <= len(model_seeds) else config_dict['seed'] + run_id
        )
        
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


def _create_config_from_dict(config_dict):
    """Helper function to create Config from dictionary"""
    config = Config()
    
    # Basic settings
    config.experiment_name = config_dict.get('experiment_name', 'experiment')
    config.seed = config_dict.get('seed', 42)
    
    # Data paths
    data_paths = config_dict.get('data_paths', {})
    config.train_path = data_paths.get('train_path')
    config.test_path = data_paths.get('test_path')
    config.augmented_data_path = data_paths.get('augmented_data_path')
    config.exclude_files_csv = data_paths.get('exclude_files_csv')
    config.data_path = os.path.dirname(data_paths.get('train_path', ''))
    if 'option2_train_path' in data_paths:
        config.option2_train_path = data_paths['option2_train_path']
    if 'option2_test_path' in data_paths:
        config.option2_test_path = data_paths['option2_test_path']
    
    # Preprocessing
    preprocessing = config_dict.get('preprocessing', {})
    config.max_series_length = preprocessing.get('max_series_length', 10000)
    config.normalize = preprocessing.get('normalize', True)
    config.include_derivatives = preprocessing.get('include_derivatives', False)
    config.validation_split = preprocessing.get('validation_split', 0.2)
    
    # Model architecture
    arch = config_dict.get('model_architecture', {})
    config.channels = arch.get('channels', [64, 128, 256, 512, 512])
    config.dilations = arch.get('dilations', [1, 2, 4, 8, 16])
    config.strides = arch.get('strides', [2, 2, 2, 2, 1])
    config.kernel_size = arch.get('kernel_size', 7)
    config.dropout = arch.get('dropout', 0.3)
    config.use_depthwise_separable = arch.get('use_depthwise_separable', False)
    config.use_residual = arch.get('use_residual', True)
    config.activation = arch.get('activation', 'swish')
    config.use_max_pooling = arch.get('use_max_pooling', True)
    config.dense_hidden_ratio = arch.get('dense_hidden_ratio', 1.0)
    config.dense_hidden_min = arch.get('dense_hidden_min', 256)
    
    # Training
    training = config_dict.get('training', {})
    config.num_epochs = training.get('num_epochs', 300)
    config.batch_size = training.get('batch_size', 64)
    config.learning_rate = training.get('learning_rate', 0.0005)
    config.weight_decay = training.get('weight_decay', 0.004)
    config.early_stopping_patience = training.get('early_stopping_patience', 25)
    config.early_stopping_metric = training.get('early_stopping_metric', 'val_loss')
    config.scheduler_factor = training.get('scheduler_factor', 0.5)
    config.scheduler_patience = training.get('scheduler_patience', 5)
    config.use_fixed_class_weights = training.get('use_fixed_class_weights', True)
    if 'fixed_class_distribution' in training:
        config.fixed_class_distribution = training['fixed_class_distribution']
    config.add_noise = training.get('add_noise', True)
    config.noise_std = training.get('noise_std', 0.08)
    config.use_masking = training.get('use_masking', False)
    
    # Output paths
    output_paths = config_dict.get('output_paths', {})
    config.model_dir = output_paths.get('models_dir', 'models')
    config.results_dir = output_paths.get('results_dir', 'results')
    config.logs_dir = output_paths.get('logs_dir', 'logs')
    
    # Status mapping
    config.status_mapping = {"Normal": 0, "NPT": 1, "OD": 2}
    config.num_classes = 3
    
    return config

