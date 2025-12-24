"""
XGBoost Classifier for EDM Drill Classification
Uses embeddings from TS2Vec foundational model to predict drill classes
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

# Add ts2vec to path first (before FINAL to avoid conflicts)
ts2vec_path = os.path.join(os.path.dirname(__file__), 'ts2vec')
sys.path.insert(0, ts2vec_path)
# Add FINAL after ts2vec
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'FINAL'))

# Import ts2vec utilities (direct imports since we added ts2vec to path)
import utils as ts2vec_utils
from ts2vec import TS2Vec
from preprocessing import EDMPreprocessor

# Alias utils functions
init_dl_program = ts2vec_utils.init_dl_program

# Import custom foundational model
try:
    from custom_foundational_model import create_custom_foundational_model
    CUSTOM_MODEL_AVAILABLE = True
except ImportError as e:
    CUSTOM_MODEL_AVAILABLE = False
    print(f"Warning: Custom foundational model not available: {e}")


def load_labeled_data(data_path, max_length=5000, apply_max_pooling=True, max_pooling_kernel_size=2, filter_classes=None):
    """
    Load labeled EDM data from class subdirectories.
    
    Args:
        data_path: Path to Train or Test folder containing class subdirectories
        max_length: Maximum length for time series
        apply_max_pooling: Whether to apply max pooling
        max_pooling_kernel_size: Kernel size for max pooling
        filter_classes: List of class names to include (None = all classes)
        
    Returns:
        data: Array of shape (n_samples, max_length, n_features)
        labels: Array of class labels (strings)
        label_encoder: Fitted LabelEncoder
    """
    print(f"Loading labeled data from {data_path}...")
    
    # Get all class subdirectories
    class_dirs = [d for d in os.listdir(data_path) 
                  if os.path.isdir(os.path.join(data_path, d))]
    
    # Filter classes if specified
    if filter_classes is not None:
        filter_classes_set = set(filter_classes)
        class_dirs = [d for d in class_dirs if d in filter_classes_set]
        print(f"Filtering to classes: {filter_classes}")
    
    class_dirs = sorted(class_dirs)
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    all_data = []
    all_labels = []
    
    preprocessor = EDMPreprocessor(max_length=max_length)
    
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        csv_files = glob(os.path.join(class_path, "*.csv"))
        
        print(f"\nProcessing class '{class_name}': {len(csv_files)} files")
        
        for csv_file in tqdm(csv_files, desc=f"  {class_name}", leave=False):
            try:
                df = pd.read_csv(csv_file)
                
                # Extract Voltage and Z columns
                if 'Voltage' not in df.columns or 'Z' not in df.columns:
                    print(f"Warning: {csv_file} missing Voltage or Z columns. Skipping.")
                    continue
                
                features = df[['Voltage', 'Z']].values.astype(np.float32)
                
                # Apply max pooling if enabled
                if apply_max_pooling:
                    features = preprocessor.max_pool(features, kernel_size=max_pooling_kernel_size)
                
                # Pad or truncate
                padded, mask = preprocessor.pad_or_truncate(features)
                
                all_data.append(padded)
                all_labels.append(class_name)
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
    
    if not all_data:
        raise ValueError(f"No valid data loaded from {data_path}")
    
    # Convert to numpy array
    data = np.array(all_data)  # (n_samples, max_length, 2)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    
    print(f"\nLoaded {len(data)} time series")
    print(f"Data shape: {data.shape}")
    print(f"Classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    print(f"Class distribution:")
    for class_name, count in pd.Series(all_labels).value_counts().items():
        print(f"  {class_name}: {count}")
    
    return data, encoded_labels, label_encoder, preprocessor


def generate_embeddings(model, data, preprocessor, batch_size=32, encoding_window='full_series', model_type='ts2vec'):
    """
    Generate embeddings for time series data using foundational model.
    
    Args:
        model: Trained foundational model (TS2Vec or CustomFoundationalEncoder)
        data: Time series data of shape (n_samples, max_length, n_features)
        preprocessor: Fitted preprocessor (for standardization)
        batch_size: Batch size for encoding
        encoding_window: Type of encoding ('full_series' for instance-level) - only for TS2Vec
        model_type: Type of model ('ts2vec' or 'custom')
        
    Returns:
        embeddings: Array of shape (n_samples, embedding_dim)
    """
    print(f"\nGenerating embeddings...")
    print(f"  Data shape: {data.shape}")
    print(f"  Model type: {model_type}")
    
    # Standardize data using preprocessor
    if not preprocessor._fitted:
        print("Fitting preprocessor on data...")
        preprocessor.fit(data)
    
    data_scaled = preprocessor.transform(data)
    
    if model_type == 'ts2vec':
        print(f"  Encoding window: {encoding_window}")
        # Generate embeddings using TS2Vec
        embeddings = model.encode(
            data_scaled,
            encoding_window=encoding_window,
            batch_size=batch_size
        )
    else:
        # Generate embeddings using custom foundational model
        import torch
        model.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for i in range(0, len(data_scaled), batch_size):
                batch_data = data_scaled[i:i+batch_size]  # (batch, length, channels)
                # Convert to (batch, channels, length) for PyTorch
                batch_tensor = torch.FloatTensor(batch_data).transpose(1, 2).to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
                batch_embeddings = model(batch_tensor)  # (batch, embedding_dim)
                embeddings_list.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.concatenate(embeddings_list, axis=0)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    
    return embeddings


def load_custom_foundational_model(model_path, config_path, device):
    """
    Load custom foundational model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        config_path: Path to config JSON file
        device: PyTorch device
        
    Returns:
        model: Loaded CustomFoundationalEncoder model
    """
    if not CUSTOM_MODEL_AVAILABLE:
        raise ImportError("Custom foundational model not available. Make sure custom_foundational_model.py exists.")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object similar to train_custom_foundational.py
    class Config:
        def __init__(self):
            pass
    
    config = Config()
    
    # Model architecture
    arch = config_dict.get('model_architecture', {})
    config.num_input_channels = 2  # Voltage and Z
    config.channels = arch.get('channels', [64, 128, 256, 512, 512])
    config.dilations = arch.get('dilations', [1, 2, 4, 8, 16])
    config.strides = arch.get('strides', [2, 2, 2, 2, 1])
    config.kernel_size = arch.get('kernel_size', 7)
    config.dropout = arch.get('dropout', 0.3)
    config.use_depthwise_separable = arch.get('use_depthwise_separable', False)
    config.use_residual = arch.get('use_residual', True)
    config.activation = arch.get('activation', 'swish')
    config.use_max_pooling = arch.get('use_max_pooling', True)
    
    # Create model
    embedding_dim = arch.get('embedding_dim', 256)
    model = create_custom_foundational_model(config, embedding_dim=embedding_dim, debug=False)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Store device for later use
    model.device = device
    
    print(f"Loaded custom foundational model from {model_path}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    
    return model


def calculate_class_weights(y_train, method='balanced', custom_weights=None):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        method: 'balanced' (inverse frequency), 'sqrt' (sqrt of inverse frequency), 
                'custom' (use custom_weights dict), or None (no weighting)
        custom_weights: Dictionary mapping class index to weight (if method='custom')
        
    Returns:
        sample_weight: Array of weights for each sample, or None
        class_weights_dict: Dictionary of class weights
    """
    if method is None:
        return None, {}
    
    from collections import Counter
    class_counts = Counter(y_train)
    n_classes = len(class_counts)
    total_samples = len(y_train)
    
    if method == 'custom' and custom_weights:
        class_weights_dict = custom_weights
    elif method == 'balanced':
        # Inverse frequency weighting
        class_weights_dict = {
            cls: total_samples / (n_classes * count) 
            for cls, count in class_counts.items()
        }
    elif method == 'sqrt':
        # Square root of inverse frequency (less aggressive)
        class_weights_dict = {
            cls: np.sqrt(total_samples / (n_classes * count))
            for cls, count in class_counts.items()
        }
    else:
        return None, {}
    
    # Create sample weights
    sample_weight = np.array([class_weights_dict[y] for y in y_train])
    
    # Normalize weights to have mean of 1.0
    sample_weight = sample_weight / sample_weight.mean()
    
    return sample_weight, class_weights_dict


def train_xgboost_classifier(X_train, y_train, X_val=None, y_val=None, config=None, 
                             class_weight_method='balanced', custom_class_weights=None):
    """
    Train XGBoost classifier on embeddings.
    
    Args:
        X_train: Training embeddings (n_samples, n_features)
        y_train: Training labels
        X_val: Validation embeddings (optional)
        y_val: Validation labels (optional)
        config: Dictionary with XGBoost hyperparameters
        class_weight_method: Method for class weighting ('balanced', 'sqrt', 'custom', or None)
        custom_class_weights: Dictionary mapping class index to weight (if class_weight_method='custom')
        
    Returns:
        model: Trained XGBoost classifier
    """
    print("\n" + "=" * 60)
    print("Training XGBoost Classifier")
    print("=" * 60)
    
    # Calculate class weights
    sample_weight, class_weights_dict = calculate_class_weights(
        y_train, method=class_weight_method, custom_weights=custom_class_weights
    )
    
    if sample_weight is not None:
        print(f"\nClass Weights Applied (method: {class_weight_method}):")
        for cls, weight in sorted(class_weights_dict.items()):
            print(f"  Class {cls}: {weight:.4f}")
        print(f"  Sample weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
    else:
        print("\nNo class weighting applied")
    
    # Default hyperparameters
    default_config = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    if config:
        default_config.update(config)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")
    print(f"\nHyperparameters:")
    for key, value in default_config.items():
        if key not in ['objective', 'eval_metric', 'verbosity']:
            print(f"  {key}: {value}")
    
    # Prepare evaluation set
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        print(f"Validation samples: {len(X_val)}")
    
    # Create and train model
    # Add early_stopping_rounds to config if validation set provided
    if eval_set is not None:
        default_config['early_stopping_rounds'] = 20
    
    model = XGBClassifier(**default_config)
    
    # Train with early stopping if validation set provided
    if eval_set is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=10
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    
    print("\nTraining completed!")
    
    return model


def evaluate_classifier(model, X_test, y_test, label_encoder, output_dir=None):
    """
    Evaluate XGBoost classifier and print metrics.
    
    Args:
        model: Trained XGBoost model
        X_test: Test embeddings
        y_test: Test labels
        label_encoder: LabelEncoder for converting back to class names
        output_dir: Directory to save results (optional)
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Calculate AUC ROC
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    
    # Binarize labels for multi-class ROC
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))
    
    # Initialize variables
    auc_roc = None
    auc_roc_per_class = {}
    macro_auc_roc = None
    micro_auc_roc = None
    valid_aucs = []
    
    # Calculate AUC ROC for each class (one-vs-rest)
    if n_classes == 2:
        # Binary classification
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"\nAUC ROC: {auc_roc:.4f}")
        auc_roc_per_class = {'class_0': None, 'class_1': auc_roc}
    else:
        # Multi-class classification
        print("\nAUC ROC (One-vs-Rest):")
        auc_roc_per_class = {}
        for i, class_name in enumerate(class_names):
            try:
                auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
                auc_roc_per_class[class_name] = auc
                print(f"  {class_name}: {auc:.4f}")
            except ValueError as e:
                # Handle case where class is not present in test set
                print(f"  {class_name}: N/A (class not in test set)")
                auc_roc_per_class[class_name] = None
        
        # Macro-averaged AUC ROC
        valid_aucs = [v for v in auc_roc_per_class.values() if v is not None]
        if valid_aucs:
            macro_auc_roc = np.mean(valid_aucs)
            print(f"\nMacro-averaged AUC ROC: {macro_auc_roc:.4f}")
        
        # Micro-averaged AUC ROC
        try:
            micro_auc_roc = roc_auc_score(y_test_binarized, y_pred_proba, average='micro', multi_class='ovr')
            print(f"Micro-averaged AUC ROC: {micro_auc_roc:.4f}")
        except Exception as e:
            print(f"Micro-averaged AUC ROC: N/A ({e})")
            micro_auc_roc = None
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Save results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Add AUC ROC to report
        if n_classes == 2:
            report['auc_roc'] = auc_roc
        else:
            report['auc_roc_per_class'] = auc_roc_per_class
            if valid_aucs:
                report['macro_auc_roc'] = macro_auc_roc
            if micro_auc_roc is not None:
                report['micro_auc_roc'] = micro_auc_roc
        
        with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save confusion matrix
        cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_label': label_encoder.inverse_transform(y_test),
            'predicted_label': label_encoder.inverse_transform(y_pred),
            'confidence': np.max(y_pred_proba, axis=1)
        })
        results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    results_dict = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }
    
    # Add AUC ROC metrics
    if n_classes == 2:
        results_dict['auc_roc'] = auc_roc
    else:
        results_dict['auc_roc_per_class'] = auc_roc_per_class
        if valid_aucs:
            results_dict['macro_auc_roc'] = macro_auc_roc
        if micro_auc_roc is not None:
            results_dict['micro_auc_roc'] = micro_auc_roc
    
    return results_dict


def tune_xgboost_hyperparameters(X_train, y_train, X_val, y_val, max_evals=50):
    """
    Tune XGBoost hyperparameters using Hyperopt.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_val: Validation embeddings
        y_val: Validation labels
        max_evals: Maximum number of hyperparameter evaluations
        
    Returns:
        best_params: Best hyperparameters found
        trials: Hyperopt trials object with all results
    """
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning with Hyperopt")
    print("=" * 60)
    print(f"Maximum evaluations: {max_evals}")
    print()
    
    # Define expanded search space for thousands of combinations
    space = {
        # Tree structure parameters - wider ranges
        'max_depth': scope.int(hp.quniform('max_depth', 2, 15, 1)),  # Expanded from 3-10 to 2-15
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 20, 1)),  # Expanded from 1-7 to 1-20
        'gamma': hp.uniform('gamma', 0, 2.0),  # Expanded from 0-0.5 to 0-2.0
        
        # Learning parameters - wider ranges
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),  # Expanded from 0.01-0.3 to 0.001-0.5
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 1000, 10)),  # Expanded from 50-300 to 50-1000
        
        # Sampling parameters - wider ranges
        'subsample': hp.uniform('subsample', 0.4, 1.0),  # Expanded from 0.6-1.0 to 0.4-1.0
        'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1.0),  # Expanded from 0.6-1.0 to 0.4-1.0
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.4, 1.0),  # Added new parameter
        'colsample_bynode': hp.uniform('colsample_bynode', 0.4, 1.0),  # Added new parameter
        
        # Regularization parameters - much wider ranges
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(100)),  # Expanded from 1e-5-10 to 1e-8-100
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(100)),  # Expanded from 1e-5-10 to 1e-8-100
        
        # Tree method and other parameters
        'tree_method': hp.choice('tree_method', ['auto', 'exact', 'approx', 'hist']),  # Added new parameter
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),  # Added new parameter
        'max_leaves': scope.int(hp.quniform('max_leaves', 0, 256, 1)),  # Added new parameter (0 means no limit)
        'max_bin': scope.int(hp.quniform('max_bin', 64, 512, 64)),  # Added new parameter
        
        # Fixed parameters
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 20
    }
    
    # Objective function
    def objective(params):
        # Convert float parameters to int where needed
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['max_leaves'] = int(params['max_leaves'])
        params['max_bin'] = int(params['max_bin'])
        
        # Handle max_leaves: 0 means no limit, but XGBoost needs None or a positive value
        if params['max_leaves'] == 0:
            params['max_leaves'] = None
        
        # Calculate class weights for this trial
        sample_weight, _ = calculate_class_weights(y_train, method='balanced')
        
        # Create and train model
        try:
            model = XGBClassifier(**params)
            
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                verbose=False
            )
            
            # Get predictions and calculate AUC
            y_pred_proba = model.predict_proba(X_val)
            
            # Calculate macro-averaged AUC ROC
            n_classes = len(np.unique(y_train))
            y_val_binarized = label_binarize(y_val, classes=range(n_classes))
            
            try:
                macro_auc = roc_auc_score(y_val_binarized, y_pred_proba, average='macro', multi_class='ovr')
            except:
                macro_auc = 0.0
        except Exception as e:
            # If training fails, return very poor score
            macro_auc = 0.0
        
        # Return negative AUC (since hyperopt minimizes)
        return {'loss': -macro_auc, 'status': STATUS_OK, 'auc': macro_auc}
    
    # Run optimization
    trials = Trials()
    
    print("Starting hyperparameter optimization...")
    print("-" * 60)
    
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=True
    )
    
    # Get best parameters
    best_params = space_eval(space, best)
    
    # Convert to proper types
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Results")
    print("=" * 60)
    
    # Display all trials sorted by AUC
    print("\nAll Trials (sorted by AUC, best first):")
    print("-" * 60)
    print(f"{'Trial':<8} {'AUC':<12} {'Max Depth':<12} {'LR':<12} {'N Est':<10} {'Subsample':<12} {'Colsample':<12}")
    print("-" * 60)
    
    # Extract results from trials
    trial_results = []
    for i, trial in enumerate(trials.trials):
        if trial['result']['status'] == STATUS_OK:
            auc = trial['result']['auc']
            params = trial['misc']['vals']
            # Get actual values (not indices)
            trial_params = {}
            for key in params:
                if len(params[key]) > 0:
                    if key in ['max_depth', 'n_estimators', 'min_child_weight']:
                        trial_params[key] = int(params[key][0])
                    else:
                        trial_params[key] = params[key][0]
            
            trial_results.append({
                'trial': i,
                'auc': auc,
                'params': trial_params
            })
    
    # Sort by AUC descending
    trial_results.sort(key=lambda x: x['auc'], reverse=True)
    
    # Display top trials with more parameters
    print(f"\nTop {min(30, len(trial_results))} Trials (sorted by AUC, best first):")
    print("-" * 120)
    print(f"{'Trial':<8} {'AUC':<12} {'Max Depth':<12} {'LR':<12} {'N Est':<10} {'Subsample':<12} {'Colsample':<12} {'Reg Alpha':<12} {'Reg Lambda':<12}")
    print("-" * 120)
    
    for result in trial_results[:30]:  # Show top 30
        params = result['params']
        print(f"{result['trial']:<8} {result['auc']:<12.6f} "
              f"{params.get('max_depth', 'N/A'):<12} "
              f"{params.get('learning_rate', 'N/A'):<12.6f} "
              f"{params.get('n_estimators', 'N/A'):<10} "
              f"{params.get('subsample', 'N/A'):<12.6f} "
              f"{params.get('colsample_bytree', 'N/A'):<12.6f} "
              f"{params.get('reg_alpha', 'N/A'):<12.6f} "
              f"{params.get('reg_lambda', 'N/A'):<12.6f}")
    
    print(f"\nBest AUC: {max([r['auc'] for r in trial_results]):.6f}")
    print(f"\nBest Hyperparameters:")
    for key, value in sorted(best_params.items()):
        if key not in ['objective', 'eval_metric', 'random_state', 'n_jobs', 'verbosity', 'early_stopping_rounds']:
            print(f"  {key}: {value}")
    
    return best_params, trials


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost classifier on TS2Vec embeddings')
    parser.add_argument('--ts2vec_model', type=str, 
                       default='ts2vec/Unlabeled__edm_run_20251119_110938/model_best.pkl',
                       help='Path to trained TS2Vec model')
    parser.add_argument('--ts2vec_config', type=str, default='ts2vec/config.json',
                       help='Path to TS2Vec config file')
    parser.add_argument('--train_data', type=str, default='../Data/Option 1/Train',
                       help='Path to training data directory')
    parser.add_argument('--test_data', type=str, default='../Data/Option 1/Test',
                       help='Path to test data directory (optional)')
    parser.add_argument('--output_dir', type=str, default='xgboost_results',
                       help='Directory to save results and model')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio (if test_data not provided)')
    parser.add_argument('--encoding_window', type=str, default='full_series',
                       choices=['full_series', None, 'multiscale'],
                       help='Type of encoding window for embeddings')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--filter_classes', type=str, nargs='+', default=None,
                       help='List of classes to include (e.g., Normal NPT OD). If None, uses all classes.')
    parser.add_argument('--tune_hyperparameters', action='store_true',
                       help='Use Hyperopt to tune hyperparameters')
    parser.add_argument('--max_evals', type=int, default=200,
                       help='Maximum number of hyperparameter evaluations (default: 200, use 1000+ for extensive search)')
    parser.add_argument('--class_weight_method', type=str, default='balanced',
                       choices=['balanced', 'sqrt', 'custom', 'none'],
                       help='Method for class weighting: balanced (inverse frequency), sqrt (sqrt of inverse), custom (use --custom_weights), or none')
    parser.add_argument('--custom_weights', type=str, nargs='+', default=None,
                       help='Custom class weights as space-separated values (e.g., "1.0 2.5 1.0" for 3 classes). Only used if --class_weight_method=custom')
    parser.add_argument('--foundational_model_type', type=str, default='ts2vec',
                       choices=['ts2vec', 'custom'],
                       help='Type of foundational model to use: ts2vec or custom')
    parser.add_argument('--custom_model_path', type=str, default=None,
                       help='Path to custom foundational model checkpoint (.pth file). Required if --foundational_model_type=custom')
    parser.add_argument('--custom_config_path', type=str, default='custom_foundational_config.json',
                       help='Path to custom foundational model config JSON file. Required if --foundational_model_type=custom')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost Classifier for EDM Drill Classification")
    print("=" * 60)
    
    # Load TS2Vec config
    ts2vec_config_path = os.path.join(os.path.dirname(__file__), args.ts2vec_config)
    with open(ts2vec_config_path, 'r') as f:
        ts2vec_config = json.load(f)
    
    print(f"\nTS2Vec Config:")
    print(f"  Max length: {ts2vec_config['max_length']}")
    print(f"  Repr dims: {ts2vec_config['repr_dims']}")
    print(f"  Apply max pooling: {ts2vec_config.get('apply_max_pooling', False)}")
    
    # Initialize device
    device = init_dl_program(args.gpu, seed=ts2vec_config.get('seed', 42))
    print(f"Using device: {device}")
    
    # Load foundational model
    if args.foundational_model_type == 'custom':
        if args.custom_model_path is None:
            raise ValueError("--custom_model_path is required when --foundational_model_type=custom")
        
        custom_model_path = os.path.join(os.path.dirname(__file__), args.custom_model_path)
        custom_config_path = os.path.join(os.path.dirname(__file__), args.custom_config_path)
        
        print(f"\nLoading custom foundational model from {custom_model_path}...")
        model = load_custom_foundational_model(custom_model_path, custom_config_path, device)
        print("Custom foundational model loaded successfully!")
        
        # Get embedding dimension from model
        embedding_dim = model.embedding_dim
    else:
        ts2vec_model_path = os.path.join(os.path.dirname(__file__), args.ts2vec_model)
        print(f"\nLoading TS2Vec model from {ts2vec_model_path}...")
        
        model = TS2Vec(
            input_dims=2,  # Voltage and Z
            output_dims=ts2vec_config['repr_dims'],
            device=device,
            batch_size=ts2vec_config['batch_size'],
            temporal_stride=ts2vec_config.get('temporal_stride', 1)
        )
        model.load(ts2vec_model_path)
        print("TS2Vec model loaded successfully!")
        
        embedding_dim = ts2vec_config['repr_dims']
    
    # Load training data
    train_data_path = os.path.join(os.path.dirname(__file__), args.train_data)
    train_data, train_labels, label_encoder, preprocessor = load_labeled_data(
        train_data_path,
        max_length=ts2vec_config['max_length'],
        apply_max_pooling=ts2vec_config.get('apply_max_pooling', False),
        max_pooling_kernel_size=ts2vec_config.get('max_pooling_kernel_size', 2),
        filter_classes=args.filter_classes
    )
    
    # Fit preprocessor on training data
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(train_data)
    
    # Generate embeddings for training data
    train_embeddings = generate_embeddings(
        model, train_data, preprocessor,
        batch_size=ts2vec_config['batch_size'],
        encoding_window=args.encoding_window,
        model_type=args.foundational_model_type
    )
    
    # Split into train/val or load test data
    if args.test_data and os.path.exists(os.path.join(os.path.dirname(__file__), args.test_data)):
        # Load test data (use same filter_classes as training)
        test_data_path = os.path.join(os.path.dirname(__file__), args.test_data)
        test_data, test_labels, _, _ = load_labeled_data(
            test_data_path,
            max_length=ts2vec_config['max_length'],
            apply_max_pooling=ts2vec_config.get('apply_max_pooling', False),
            max_pooling_kernel_size=ts2vec_config.get('max_pooling_kernel_size', 2),
            filter_classes=args.filter_classes
        )
        
        # Generate embeddings for test data
        test_embeddings = generate_embeddings(
            model, test_data, preprocessor,
            batch_size=ts2vec_config['batch_size'],
            encoding_window=args.encoding_window,
            model_type=args.foundational_model_type
        )
        
        X_train, y_train = train_embeddings, train_labels
        X_val, y_val = test_embeddings, test_labels
        
        print(f"\nUsing separate test set: {len(X_val)} samples")
    else:
        # Split training data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            train_embeddings, train_labels,
            test_size=args.validation_split,
            random_state=42,
            stratify=train_labels
        )
        print(f"\nSplit training data: {len(X_train)} train, {len(X_val)} validation")
    
    # Tune or train XGBoost
    if args.tune_hyperparameters:
        # Hyperparameter tuning
        best_params, trials = tune_xgboost_hyperparameters(
            X_train, y_train, X_val, y_val, 
            max_evals=args.max_evals
        )
        
        # Parse custom weights if provided
        custom_weights_dict = None
        if args.class_weight_method == 'custom' and args.custom_weights:
            try:
                weights = [float(w) for w in args.custom_weights]
                unique_classes = sorted(np.unique(y_train))
                if len(weights) == len(unique_classes):
                    custom_weights_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}
                    print(f"\nUsing custom class weights: {custom_weights_dict}")
                else:
                    print(f"Warning: Number of custom weights ({len(weights)}) doesn't match number of classes ({len(unique_classes)}). Using balanced weights instead.")
                    args.class_weight_method = 'balanced'
            except ValueError:
                print("Warning: Invalid custom weights format. Using balanced weights instead.")
                args.class_weight_method = 'balanced'
        
        # Train final model with best parameters
        print("\n" + "=" * 60)
        print("Training Final Model with Best Hyperparameters")
        print("=" * 60)
        xgb_model = train_xgboost_classifier(
            X_train, y_train, X_val, y_val, 
            config=best_params,
            class_weight_method=args.class_weight_method if args.class_weight_method != 'none' else None,
            custom_class_weights=custom_weights_dict
        )
        
        # Save tuning results
        tuning_results = {
            'best_params': best_params,
            'best_auc': max([t['result']['auc'] for t in trials.trials if t['result']['status'] == STATUS_OK]),
            'n_trials': len(trials.trials),
            'all_trials': [
                {
                    'trial': i,
                    'auc': t['result']['auc'],
                    'params': {k: v[0] if isinstance(v, list) and len(v) > 0 else v 
                              for k, v in t['misc']['vals'].items()}
                }
                for i, t in enumerate(trials.trials) 
                if t['result']['status'] == STATUS_OK
            ]
        }
        
        # Save tuning results
        tuning_output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
        os.makedirs(tuning_output_dir, exist_ok=True)
        with open(os.path.join(tuning_output_dir, 'hyperparameter_tuning_results.json'), 'w') as f:
            json.dump(tuning_results, f, indent=2, default=str)
        print(f"\nHyperparameter tuning results saved to {tuning_output_dir}/hyperparameter_tuning_results.json")
    else:
        # Parse custom weights if provided
        custom_weights_dict = None
        if args.class_weight_method == 'custom' and args.custom_weights:
            try:
                weights = [float(w) for w in args.custom_weights]
                unique_classes = sorted(np.unique(y_train))
                if len(weights) == len(unique_classes):
                    custom_weights_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}
                    print(f"\nUsing custom class weights: {custom_weights_dict}")
                else:
                    print(f"Warning: Number of custom weights ({len(weights)}) doesn't match number of classes ({len(unique_classes)}). Using balanced weights instead.")
                    args.class_weight_method = 'balanced'
            except ValueError:
                print("Warning: Invalid custom weights format. Using balanced weights instead.")
                args.class_weight_method = 'balanced'
        
        # Standard training
        xgb_model = train_xgboost_classifier(
            X_train, y_train, X_val, y_val,
            class_weight_method=args.class_weight_method if args.class_weight_method != 'none' else None,
            custom_class_weights=custom_weights_dict
        )
    
    # Evaluate on validation/test set
    results = evaluate_classifier(xgb_model, X_val, y_val, label_encoder, args.output_dir)
    
    # Save model and metadata
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save XGBoost model
    model_path = os.path.join(output_dir, 'xgboost_model.json')
    xgb_model.save_model(model_path)
    print(f"\nXGBoost model saved to {model_path}")
    
    # Save label encoder
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save metadata
    metadata = {
        'ts2vec_model_path': args.ts2vec_model,
        'ts2vec_config': ts2vec_config,
        'encoding_window': args.encoding_window,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'embedding_dim': train_embeddings.shape[1],
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'accuracy': float(results['accuracy'])
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - xgboost_model.json: Trained XGBoost model")
    print(f"  - label_encoder.pkl: Label encoder")
    print(f"  - metadata.json: Training metadata")
    print(f"  - classification_report.json: Classification metrics")
    print(f"  - confusion_matrix.csv: Confusion matrix")
    print(f"  - predictions.csv: Test predictions")


if __name__ == '__main__':
    main()

