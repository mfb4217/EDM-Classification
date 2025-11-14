"""
Configuration loader for status classification experiments
Loads configuration from config.json and provides Config object
"""
import json
import os


class Config:
    """Experiment configuration object - loaded from config.json"""
    
    def __init__(self):
        """Initialize empty config - must be loaded from JSON"""
        pass
    
    @property
    def num_input_channels(self) -> int:
        """Number of input channels: 2 (Voltage, Z) or 4 (with derivatives)"""
        return 4 if getattr(self, 'include_derivatives', False) else 2
    
    @property
    def receptive_field(self) -> int:
        """Calculate receptive field based on dilations and kernel_size"""
        rf = 1
        dilations = getattr(self, 'dilations', [])
        kernel_size = getattr(self, 'kernel_size', 7)
        for dilation in dilations:
            rf += (kernel_size - 1) * dilation
        return rf
    
    def get_class_names(self):
        """Get list of class names"""
        status_mapping = getattr(self, 'status_mapping', {"Normal": 0, "NPT": 1, "OD": 2})
        return list(status_mapping.keys())
    
    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return 3
    
    @property
    def status_mapping(self) -> dict:
        """Status mapping (if not set, use default)"""
        return getattr(self, '_status_mapping', {"Normal": 0, "NPT": 1, "OD": 2})


def load_config_from_json(json_path: str) -> Config:
    """
    Load configuration from config.json and convert to Config object.
    
    This function converts the nested JSON structure to a flat Config object
    with all attributes accessible as object properties.
    
    Args:
        json_path: Path to config.json file
        
    Returns:
        Config object with all settings loaded
    """
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    return dict_to_config(config_dict)


def dict_to_config(config_dict: dict) -> Config:
    """
    Convert configuration dictionary (from JSON) to Config object.
    
    Flattens the nested structure from config.json into a flat Config object.
    
    Args:
        config_dict: Dictionary from config.json
        
    Returns:
        Config object
    """
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
    config.mask_ratio = training.get('mask_ratio', 0.1)
    
    # Output paths
    output_paths = config_dict.get('output_paths', {})
    config.model_dir = output_paths.get('models_dir', 'models')
    config.results_dir = output_paths.get('results_dir', 'results')
    config.logs_dir = output_paths.get('logs_dir', 'logs')
    
    # Status mapping (always the same)
    config._status_mapping = {"Normal": 0, "NPT": 1, "OD": 2}
    
    return config


def create_config(base_dict: dict, **kwargs) -> Config:
    """
    Create a Config object from a base dictionary with optional overrides.
    
    Used for creating variations of configurations (e.g., different seeds).
    
    Args:
        base_dict: Base configuration dictionary
        **kwargs: Override values
        
    Returns:
        Config object
    """
    config_dict = base_dict.copy()
    config_dict.update(kwargs)
    return dict_to_config(config_dict)
