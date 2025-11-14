"""
Inference Module
Make predictions on a single drill using trained ensemble
"""
import os
import torch
import numpy as np
import pandas as pd
import joblib
from config import Config
from preprocessing import DataPreprocessor
from utils import preprocess_single_drill, predict_ensemble_production, apply_thresholds
from thresholds import load_thresholds, apply_thresholds_selective


def save_preprocessor(preprocessor, output_path):
    """
    Save preprocessor (scaler) to file.
    
    Args:
        preprocessor: DataPreprocessor instance with fitted scaler
        output_path: Path to save the preprocessor
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(preprocessor.scaler, output_path)
    print(f"Preprocessor (scaler) saved to: {output_path}")


def load_preprocessor(input_path, config):
    """
    Load preprocessor (scaler) from file.
    
    Args:
        input_path: Path to saved scaler
        config: Config object
        
    Returns:
        DataPreprocessor instance with loaded scaler
    """
    from preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(config)
    if os.path.exists(input_path):
        preprocessor.scaler = joblib.load(input_path)
        print(f"Preprocessor (scaler) loaded from: {input_path}")
    else:
        print(f"Warning: Preprocessor file not found: {input_path}")
        print("Fitting scaler on training data...")
        preprocessor.preprocess_train(config.train_path)
    
    return preprocessor


def load_ensemble_models(config_dict, model_dir=None):
    """
    Load trained ensemble models.
    
    Args:
        config_dict: Configuration dictionary
        model_dir: Directory containing model checkpoints (default: results/{experiment_name}_model_*/models/)
        
    Returns:
        models: List of loaded models in inference mode
        config: Config object
        preprocessor: Fitted preprocessor with loaded scaler
    """
    from model import create_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create config
    config = _create_config_from_dict(config_dict)
    
    # Find model directories
    if model_dir is None:
        results_dir = config_dict['output_paths']['results_dir']
        experiment_name = config_dict['experiment_name']
        model_dirs = []
        for i in range(1, config_dict['ensemble']['num_models'] + 1):
            # Models are saved directly in model directory (no nested models/ folder)
            model_path = os.path.join(results_dir, f"{experiment_name}_model_{i:02d}", 
                                     "best_model.pth")
            if os.path.exists(model_path):
                model_dirs.append(model_path)
    else:
        # Load from specific directory
        model_dirs = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_dirs:
        raise ValueError(f"No model checkpoints found!")
    
    # Load models
    models = []
    for model_path in model_dirs:
        model = create_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        print(f"Loaded model: {os.path.basename(model_path)}")
    
    # Load preprocessor with saved scaler (CRITICAL: don't refit!)
    results_dir = config_dict['output_paths']['results_dir']
    experiment_name = config_dict['experiment_name']
    scaler_path = os.path.join(results_dir, f"{experiment_name}_scaler.pkl")
    
    preprocessor = load_preprocessor(scaler_path, config)
    
    return models, config, preprocessor


def predict_single_drill(csv_file_path, models, preprocessor, thresholds, device=None, class_names=None):
    """
    Predict status for a single drill.
    
    Args:
        csv_file_path: Path to CSV file with drill data
        models: List of trained models
        preprocessor: Fitted preprocessor (with loaded scaler, NOT refitted)
        thresholds: Array of thresholds (or path to JSON file)
        device: PyTorch device (default: CPU)
        class_names: List of class names (default: ["Normal", "NPT", "OD"])
        
    Returns:
        Dictionary with:
            - prediction: Predicted class index
            - class_name: Predicted class name
            - probabilities: Array of probabilities per class
            - probabilities_dict: Dictionary with probabilities per class
    """
    if device is None:
        device = torch.device('cpu')
    
    if class_names is None:
        class_names = ["Normal", "NPT", "OD"]
    
    # Load thresholds if path provided
    if isinstance(thresholds, str):
        thresholds = load_thresholds(thresholds, class_names)
    
    # Preprocess drill
    x_tensor = preprocess_single_drill(preprocessor, csv_file_path, 
                                       max_length=preprocessor.config.max_series_length)
    
    # Get ensemble prediction
    avg_probs = predict_ensemble_production(models, x_tensor, device)
    
    # Apply thresholds
    # Use selective threshold application (argmax for Normal/OD, threshold for NPT)
    class_names_inference = ["Normal", "NPT", "OD"]
    prediction_idx = apply_thresholds_selective(avg_probs.reshape(1, -1), thresholds, 
                                                class_names_inference, optimize_only_npt=True)[0]
    prediction_name = class_names[prediction_idx]
    
    # Create probabilities dictionary
    probabilities_dict = {name: float(avg_probs[i]) for i, name in enumerate(class_names)}
    
    return {
        'prediction': int(prediction_idx),
        'class_name': prediction_name,
        'probabilities': avg_probs.tolist(),
        'probabilities_dict': probabilities_dict
    }


def _create_config_from_dict(config_dict):
    """Helper function to create Config from dictionary"""
    from config import dict_to_config
    return dict_to_config(config_dict)
