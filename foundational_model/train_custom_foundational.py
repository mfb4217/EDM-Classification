"""
Training script for custom foundational model
Trains encoder with self-supervised contrastive learning on unlabeled data
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from datetime import datetime

# Add paths - ts2vec first to avoid conflicts
ts2vec_path = os.path.join(os.path.dirname(__file__), 'ts2vec')
sys.path.insert(0, ts2vec_path)
# Add FINAL after ts2vec for custom model imports
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'FINAL'))

# Import ts2vec utilities (direct imports since we added ts2vec to path)
import utils as ts2vec_utils
from edm_dataloader import load_edm_unlabeled, split_data_into_chunks

# Import custom foundational model components
from custom_foundational_model import create_custom_foundational_model, count_parameters
from contrastive_losses import hierarchical_contrastive_loss, hierarchical_contrastive_loss_soft, compute_soft_labels_from_data

# Import FINAL config (only for dict_to_config if needed, but we're using our own config loader)
try:
    from FINAL.config import dict_to_config
except ImportError:
    dict_to_config = None

# Alias utils functions
init_dl_program = ts2vec_utils.init_dl_program
name_with_datetime = ts2vec_utils.name_with_datetime


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with data augmentation"""
    
    def __init__(self, X, config, is_training=True):
        """
        Args:
            X: Preprocessed data array of shape (n_samples, max_length, num_channels)
            config: Configuration object
            is_training: Whether this is training data (enables augmentation)
        """
        self.X = torch.FloatTensor(X)  # (n_samples, max_length, num_channels)
        self.config = config
        self.is_training = is_training
        self.max_length = X.shape[1]
        
        # Augmentation config
        self.add_noise = getattr(config, 'add_noise', True)
        self.noise_std = getattr(config, 'noise_std', 0.08)
        self.random_crop = getattr(config, 'random_crop', True)
        self.crop_min_length = getattr(config, 'crop_min_length', 1000)
        self.crop_max_length = getattr(config, 'crop_max_length', 5000)
        
        if self.add_noise:
            self.feature_std = torch.std(self.X, dim=(0, 1), keepdim=True)
    
    def __len__(self):
        return len(self.X)
    
    def _augment(self, x):
        """Apply data augmentation to a single time series"""
        # Convert to (num_channels, length) for processing
        x = x.transpose(0, 1)  # (num_channels, length)
        
        if not self.is_training:
            return x
        
        # Random crop (similar to TS2Vec)
        if self.random_crop and x.shape[1] > self.crop_min_length:
            crop_length = random.randint(self.crop_min_length, min(self.crop_max_length, x.shape[1]))
            start_idx = random.randint(0, x.shape[1] - crop_length)
            x = x[:, start_idx:start_idx + crop_length]
            
            # Pad back to max_length if needed
            if x.shape[1] < self.max_length:
                padding = self.max_length - x.shape[1]
                x = torch.nn.functional.pad(x, (0, padding), mode='constant', value=0)
            elif x.shape[1] > self.max_length:
                x = x[:, :self.max_length]
        
        # Add noise
        if self.add_noise:
            noise = torch.randn_like(x) * self.noise_std
            noise_scaled = noise * self.feature_std.squeeze()[:, None]
            x = x + noise_scaled
        
        return x
    
    def __getitem__(self, idx):
        x = self.X[idx]  # (max_length, num_channels)
        
        # Generate two augmented views
        x1 = self._augment(x)
        x2 = self._augment(x)
        
        return x1, x2


def create_config_from_json(json_path):
    """Load config from JSON and convert to Config object"""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create a simple config object
    class Config:
        def __init__(self):
            pass
    
    config = Config()
    
    # Set all config values
    config.seed = config_dict.get('seed', 42)
    config.data_paths = config_dict.get('data_paths', {})
    
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
    
    # Training
    training = config_dict.get('training', {})
    config.num_epochs = training.get('num_epochs', 100)
    config.batch_size = training.get('batch_size', 16)
    config.learning_rate = training.get('learning_rate', 0.001)
    config.weight_decay = training.get('weight_decay', 0.0001)
    config.early_stopping_patience = training.get('early_stopping_patience', 20)
    config.scheduler_factor = training.get('scheduler_factor', 0.5)
    config.scheduler_patience = training.get('scheduler_patience', 5)
    config.save_every = training.get('save_every', 20)
    
    # Contrastive loss config
    if 'contrastive_loss' in config_dict:
        for key, value in config_dict['contrastive_loss'].items():
            setattr(config, key, value)
    else:
        config.alpha = 0.5
        config.temporal_unit = 0
        config.temporal_stride = 4
    
    # Data augmentation config
    if 'data_augmentation' in config_dict:
        for key, value in config_dict['data_augmentation'].items():
            setattr(config, key, value)
    else:
        config.add_noise = True
        config.noise_std = 0.08
        config.random_crop = True
        config.crop_min_length = 1000
        config.crop_max_length = 5000
    
    # Preprocessing config
    if 'preprocessing' in config_dict:
        config.preprocessing = config_dict['preprocessing']
    else:
        config.preprocessing = {}
    
    # Output paths
    if 'output_paths' in config_dict:
        config.output_paths = config_dict['output_paths']
    else:
        config.output_paths = {}
    
    return config


def train_epoch(model, train_loader, optimizer, device, config, soft_labels=None, batch_indices_map=None):
    """Train for one epoch using SoftCLT loss"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    # Get SoftCLT config
    use_softclt = getattr(config, 'use_softclt', True)
    soft_temporal = getattr(config, 'soft_temporal', True)
    soft_instance = getattr(config, 'soft_instance', False)
    tau_temp = getattr(config, 'tau_temp', 2.0)
    lambda_ = getattr(config, 'lambda_', 0.5)
    temporal_unit = getattr(config, 'temporal_unit', 0)
    temporal_stride = getattr(config, 'temporal_stride', 4)
    
    for batch_idx, (x1, x2) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x1 = x1.to(device)  # (batch, num_channels, length)
        x2 = x2.to(device)  # (batch, num_channels, length)
        
        if use_softclt:
            # Get temporal features for SoftCLT
            _, z1_temporal = model(x1, return_temporal=True)  # (batch, channels, temporal_length)
            _, z2_temporal = model(x2, return_temporal=True)  # (batch, channels, temporal_length)
            
            # Convert to (B, T, C) format for SoftCLT
            z1_temporal = z1_temporal.transpose(1, 2)  # (batch, temporal_length, channels)
            z2_temporal = z2_temporal.transpose(1, 2)  # (batch, temporal_length, channels)
            
            # Get soft labels for this batch if available
            batch_soft_labels = None
            if soft_instance and soft_labels is not None and batch_indices_map is not None:
                batch_indices = batch_indices_map[batch_idx] if batch_idx < len(batch_indices_map) else None
                if batch_indices is not None:
                    batch_soft_labels = soft_labels[batch_indices][:, batch_indices]  # (B, B)
            
            # Compute SoftCLT loss
            loss = hierarchical_contrastive_loss_soft(
                z1_temporal, z2_temporal,
                soft_labels=batch_soft_labels,
                tau_temp=tau_temp,
                lambda_=lambda_,
                temporal_unit=temporal_unit,
                soft_temporal=soft_temporal,
                soft_instance=(batch_soft_labels is not None),
                temporal_hierarchy=True,
                temporal_stride=temporal_stride
            )
        else:
            # Fallback to original loss (for compatibility)
            z1 = model(x1)  # (batch, embedding_dim)
            z2 = model(x2)  # (batch, embedding_dim)
            
            # Normalize embeddings
            z1 = nn.functional.normalize(z1, p=2, dim=1)
            z2 = nn.functional.normalize(z2, p=2, dim=1)
            
            # Compute instance contrastive loss
            batch_size = z1.size(0)
            sim_matrix = torch.matmul(z1, z2.t())  # (batch, batch)
            labels = torch.arange(batch_size, device=z1.device)
            loss = nn.functional.cross_entropy(sim_matrix / 0.07, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(model, val_loader, device, config, soft_labels=None, batch_indices_map=None):
    """Validate model using SoftCLT loss"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    # Get SoftCLT config
    use_softclt = getattr(config, 'use_softclt', True)
    soft_temporal = getattr(config, 'soft_temporal', True)
    soft_instance = getattr(config, 'soft_instance', False)
    tau_temp = getattr(config, 'tau_temp', 2.0)
    lambda_ = getattr(config, 'lambda_', 0.5)
    temporal_unit = getattr(config, 'temporal_unit', 0)
    temporal_stride = getattr(config, 'temporal_stride', 4)
    
    with torch.no_grad():
        for batch_idx, (x1, x2) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            if use_softclt:
                # Get temporal features for SoftCLT
                _, z1_temporal = model(x1, return_temporal=True)
                _, z2_temporal = model(x2, return_temporal=True)
                
                # Convert to (B, T, C) format
                z1_temporal = z1_temporal.transpose(1, 2)
                z2_temporal = z2_temporal.transpose(1, 2)
                
                # Get soft labels for this batch if available
                batch_soft_labels = None
                if soft_instance and soft_labels is not None and batch_indices_map is not None:
                    batch_indices = batch_indices_map[batch_idx] if batch_idx < len(batch_indices_map) else None
                    if batch_indices is not None:
                        batch_soft_labels = soft_labels[batch_indices][:, batch_indices]
                
                # Compute SoftCLT loss
                loss = hierarchical_contrastive_loss_soft(
                    z1_temporal, z2_temporal,
                    soft_labels=batch_soft_labels,
                    tau_temp=tau_temp,
                    lambda_=lambda_,
                    temporal_unit=temporal_unit,
                    soft_temporal=soft_temporal,
                    soft_instance=(batch_soft_labels is not None),
                    temporal_hierarchy=True,
                    temporal_stride=temporal_stride
                )
            else:
                # Fallback to original loss
                z1 = model(x1)
                z2 = model(x2)
                z1 = nn.functional.normalize(z1, p=2, dim=1)
                z2 = nn.functional.normalize(z2, p=2, dim=1)
                batch_size = z1.size(0)
                sim_matrix = torch.matmul(z1, z2.t())
                labels = torch.arange(batch_size, device=z1.device)
                loss = nn.functional.cross_entropy(sim_matrix / 0.07, labels)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom foundational model')
    parser.add_argument('--config', type=str, default='custom_foundational_config.json',
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = create_config_from_json(config_path)
    
    print("=" * 60)
    print("Custom Foundational Model Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print()
    
    # Initialize device
    device = init_dl_program(args.gpu, seed=config.seed)
    print(f"Using device: {device}")
    
    # Load unlabeled data
    unlabeled_path = os.path.join(os.path.dirname(__file__), config.data_paths.get('unlabeled_data_path', 'Unlabeled'))
    print(f"\nLoading unlabeled data from {unlabeled_path}...")
    
    preprocessing_config = config.preprocessing if hasattr(config, 'preprocessing') else {}
    train_data, preprocessor = load_edm_unlabeled(
        unlabeled_path,
        max_length=preprocessing_config.get('max_series_length', 5000),
        apply_max_pooling=preprocessing_config.get('apply_max_pooling', False),
        max_pooling_kernel_size=preprocessing_config.get('max_pooling_kernel_size', 2)
    )
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_data_split, val_data_split = train_test_split(
        train_data,
        test_size=preprocessing_config.get('validation_split', 0.2),
        random_state=config.seed
    )
    
    print(f"Train samples: {len(train_data_split)}")
    print(f"Validation samples: {len(val_data_split)}")
    
    # Compute soft labels if soft instance contrastive learning is enabled
    soft_labels = None
    train_batch_indices_map = None
    val_batch_indices_map = None
    
    use_softclt = getattr(config, 'use_softclt', True)
    soft_instance = getattr(config, 'soft_instance', False)
    
    if use_softclt and soft_instance:
        print("\nComputing soft labels for instance contrastive learning...")
        dist_type = getattr(config, 'dist_type', 'euclidean')
        try:
            soft_labels = compute_soft_labels_from_data(
                train_data_split,
                dist_type=dist_type,
                min_val=0,
                max_val=1
            )
            print(f"Soft labels computed: shape {soft_labels.shape}")
            print(f"Soft label range: [{soft_labels.min():.4f}, {soft_labels.max():.4f}]")
        except Exception as e:
            print(f"Warning: Could not compute soft labels: {e}")
            print("Falling back to hard instance contrastive learning")
            soft_instance = False
            config.soft_instance = False
    
    # Create datasets
    train_dataset = ContrastiveDataset(train_data_split, config, is_training=True)
    val_dataset = ContrastiveDataset(val_data_split, config, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Create batch indices map for soft labels (if needed)
    # Note: For simplicity, we'll skip batch mapping for now
    # Soft instance CL requires exact batch indices, which is complex with shuffling
    # For now, we'll use soft temporal CL only (which doesn't need batch mapping)
    if soft_labels is not None and soft_instance:
        print("Note: Soft instance CL requires batch index mapping.")
        print("For now, using soft temporal CL only. Set soft_instance=False in config.")
        soft_labels = None  # Disable soft instance CL for now
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    
    # Get embedding dimension from config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    arch_config = config_dict.get('model_architecture', {})
    embedding_dim = arch_config.get('embedding_dim', 256)
    
    model = create_custom_foundational_model(
        config,
        embedding_dim=embedding_dim,
        debug=False
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Create output directory
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    experiment_name = config_dict.get('experiment_name', 'custom_foundational')
    run_name = name_with_datetime(experiment_name)
    output_paths = config_dict.get('output_paths', {})
    model_dir = os.path.join(os.path.dirname(__file__), output_paths.get('model_dir', 'custom_foundational_models'), run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config, 
                                soft_labels=soft_labels, batch_indices_map=train_batch_indices_map)
        
        # Validate
        val_loss = validate(model, val_loader, device, config, 
                           soft_labels=soft_labels, batch_indices_map=val_batch_indices_map)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(model_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__ if hasattr(config, '__dict__') else None
            }, best_model_path)
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        # Periodic save
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print("Training Completed")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_dir}")


if __name__ == '__main__':
    main()

