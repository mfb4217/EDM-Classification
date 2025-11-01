"""
Configuration for status classification experiments
Classify whole series as Normal, NPT, or OD
"""
from typing import List

class Config:
    """Experiment configuration"""
    
    # Experiment info
    experiment_name = "status_exp_001"
    seed = 42
    
    # Preprocessing
    max_series_length = 10000  # Maximum series length (pad shorter, truncate longer)
    normalize = True  # Standardize series
    include_derivatives = True  # If True, add dV/dt and dZ/dt as additional features
    
    # Data augmentation
    add_noise = True  # If True, add Gaussian noise during training
    noise_std = 0.1  # Standard deviation of Gaussian noise (as fraction of feature std)
    use_masking = False  # If True, randomly mask parts of the series during training
    mask_ratio = 0.1  # Fraction of timesteps to mask if use_masking is True
    
    # Classes to predict (Normal, NPT, OD)
    status_mapping = {"Normal": 0, "NPT": 1, "OD": 2}
    num_classes = 3
    
    # Fixed class distribution for weighting (to avoid imbalance from synthetic data)
    # These represent the desired distribution: NPT=66, Normal=220, OD=75
    use_fixed_class_weights = False  # If True, use fixed weights instead of computing from data
    fixed_class_distribution = {"Normal": 220, "NPT": 66, "OD": 75}  # Expected class counts
    
    # Architecture hyperparameters
    channels = [64, 128, 256, 512, 512]  # Much larger network
    dilations = [1, 2, 4, 8, 16]  # Match channels length
    strides = [2, 2, 2, 2, 1]  # Strides to reduce length progressively
    kernel_size = 7  # Larger kernel
    dropout = 0.35  # Heavy dropout for regularization
    use_depthwise_separable = False  # Standard conv for maximum capacity
    use_residual = True  # Use residual connections
    activation = "swish"  # Swish activation (SOTA)
    
    # Classifier head hyperparameters
    dense_hidden_ratio = 1.0  # Larger dense layers
    dense_hidden_min = 256  # Larger minimum
    use_max_pooling = True  # Use both avg and max pooling (SOTA)
    
    # Training hyperparameters
    batch_size = 64  # Reduced batch size for more updates
    learning_rate = 5e-4  # Higher learning rate for large network
    weight_decay = 5e-3  # Heavier weight decay
    num_epochs = 300
    early_stopping_patience = 50  # Much more patience
    early_stopping_metric = "val_loss"  # Use validation loss (prevents overfitting)
    
    # Learning rate scheduler
    scheduler_factor = 0.5
    scheduler_patience = 5
    
    # Data paths (Option 1)
    data_path = "../Data/Option 1"
    train_path = "../Data/Option 1/Train"
    test_path = "../Data/Option 1/Test"
    
    # Output paths
    model_dir = "models"
    results_dir = "results"
    logs_dir = "logs"
    
    @property
    def num_input_channels(self) -> int:
        """Number of input channels: 2 (Voltage, Z) or 4 (with derivatives)"""
        return 4 if self.include_derivatives else 2
    
    @property
    def receptive_field(self) -> int:
        """Calculate receptive field based on dilations and kernel_size"""
        rf = 1
        for dilation in self.dilations:
            rf += (self.kernel_size - 1) * dilation
        return rf
    
    def get_class_names(self):
        """Get list of class names"""
        return list(self.status_mapping.keys())
    
    def save_config(self, path: str):
        """Save configuration to file"""
        import json
        config_dict = {}
        for key in dir(self):
            if not key.startswith('_') and key not in ['save_config', 'load_config', 'num_classes', 'num_input_channels', 'receptive_field']:
                value = getattr(self, key)
                if not callable(value):
                    config_dict[key] = value
        config_dict['num_classes'] = self.num_classes
        config_dict['num_input_channels'] = self.num_input_channels
        config_dict['receptive_field'] = self.receptive_field
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if key not in ['num_classes', 'num_input_channels', 'receptive_field']:
                setattr(config, key, value)
        return config

