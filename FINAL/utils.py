"""
Utility functions for status classification pipeline
Contains functions from run_ensemble.py and evaluate_runtime.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Default limits
FPR_LIMITS = {'Normal': 0.15, 'NPT': 0.05, 'OD': 0.05}
FNR_LIMITS = {'Normal': 0.10, 'NPT': 0.30, 'OD': 0.20}


def apply_thresholds(probabilities, thresholds):
    """Apply per-class probability thresholds to generate predictions."""
    predictions = []
    for probs in probabilities:
        sorted_indices = np.argsort(probs)[::-1]  # Descending order
        assigned = None
        for idx in sorted_indices:
            if probs[idx] >= thresholds[idx]:
                assigned = idx
                break
        if assigned is None:
            assigned = sorted_indices[0]
        predictions.append(assigned)
    return np.array(predictions, dtype=int)


def compute_class_rates(cm):
    """Compute per-class false positive and false negative rates from confusion matrix."""
    num_classes = cm.shape[0]
    fpr = []
    fnr = []
    total = cm.sum()
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        
        fpr_den = fp + tn
        fnr_den = fn + tp
        
        fpr.append(fp / fpr_den if fpr_den > 0 else 0.0)
        fnr.append(fn / fnr_den if fnr_den > 0 else 0.0)
    return np.array(fpr), np.array(fnr)


def evaluate_with_thresholds(probabilities, y_true, thresholds, class_names):
    """Evaluate ensemble predictions with custom thresholds."""
    y_pred = apply_thresholds(probabilities, thresholds)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fpr, fnr = compute_class_rates(cm)
    
    try:
        roc_auc = roc_auc_score(y_true, probabilities, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fpr': fpr.tolist(),
        'fnr': fnr.tolist(),
        'confusion_matrix': cm.tolist()
    }


def check_constraints(fpr, fnr, class_names):
    """Check if FPR/FNR meet predefined limits for each class."""
    for idx, class_name in enumerate(class_names):
        if fpr[idx] > FPR_LIMITS[class_name] or fnr[idx] > FNR_LIMITS[class_name]:
            return False
    return True


def preprocess_single_drill(preprocessor, csv_file_path, max_length=10000):
    """Preprocess a single drill CSV file as in production"""
    df = pd.read_csv(csv_file_path)
    
    # Extract features (Voltage, Z)
    features = df[['Voltage', 'Z']].values.astype(np.float32)
    
    # Add derivatives if configured
    if preprocessor.config.include_derivatives:
        dV = np.gradient(features[:, 0])
        dZ = np.gradient(features[:, 1])
        features = np.column_stack([features, dV, dZ]).astype(np.float32)
    
    # Pad or truncate
    current_length = len(features)
    if current_length >= max_length:
        padded = features[:max_length]
    else:
        padding_length = max_length - current_length
        padded = np.pad(features, ((padding_length, 0), (0, 0)), mode='constant', constant_values=0)
    
    # Normalize with fitted scaler
    if preprocessor.config.normalize:
        num_channels = padded.shape[1]
        padded_reshaped = padded.reshape(-1, num_channels)
        padded_scaled = preprocessor.scaler.transform(padded_reshaped)
        padded = padded_scaled.reshape(padded.shape).astype(np.float32)
    
    # Convert to tensor format (num_channels, length)
    x_tensor = torch.FloatTensor(padded).transpose(0, 1)  # (num_channels, length)
    
    return x_tensor


def predict_ensemble_production(models, x_tensor, device):
    """Get predictions from all models for a single drill (batch size = 1, inference mode)"""
    # Process one drill at a time: (num_channels, length) -> (1, num_channels, length)
    x_tensor = x_tensor.unsqueeze(0).to(device)  # Batch size = 1
    
    all_probs = []
    # Inference mode: no gradients, models in eval() mode (no dropout)
    with torch.no_grad():
        for model in models:
            # Ensure model is in eval mode (disables dropout)
            model.eval()
            logits = model(x_tensor)  # (1, num_classes)
            probs = torch.softmax(logits, dim=1)  # (1, num_classes)
            all_probs.append(probs.cpu().numpy()[0])  # Extract single sample: (num_classes,)
            # Free memory immediately for CPU with limited RAM
            del logits, probs
    
    # Clean up tensor to free memory
    del x_tensor
    
    # Average probabilities across ensemble
    avg_probs = np.mean(all_probs, axis=0)  # (num_classes,)
    return avg_probs


def create_config(base_config, **kwargs):
    """Create a new config with overridden parameters (legacy compatibility)"""
    from config import dict_to_config
    # base_config can be dict or Config object
    if hasattr(base_config, '__dict__'):
        config_dict = base_config.__dict__.copy()
    else:
        config_dict = base_config.copy()
    # Override with kwargs
    config_dict.update(kwargs)
    return dict_to_config(config_dict)

