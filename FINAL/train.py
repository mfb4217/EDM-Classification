"""
Training script for status classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
from config import Config
from model import create_model, count_parameters
from preprocessing import DataPreprocessor

class ChunkDataset(Dataset):
    """Dataset for complete series with status labels and padding masks"""
    
    def __init__(self, X, y, masks=None, config=None, is_training=True):
        self.X = torch.FloatTensor(X)  # (n_series, length, num_channels)
        self.y = torch.LongTensor(y)   # (n_series,) - scalar labels!
        self.masks = torch.FloatTensor(masks) if masks is not None else None  # (n_series, length)
        self.config = config
        self.is_training = is_training
        
        if config is not None and config.add_noise:
            self.feature_std = torch.std(self.X, dim=(0, 1), keepdim=True)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert to (num_channels, length)
        x = self.X[idx].transpose(0, 1)
        y = self.y[idx].item()  # Scalar
        
        # Data augmentation during training
        if self.config is not None and self.is_training:
            # Random masking
            if getattr(self.config, 'use_masking', False):
                mask = self.masks[idx] if self.masks is not None else torch.ones(x.shape[1])
                num_timesteps = mask.sum().int().item()
                num_to_mask = int(num_timesteps * self.config.mask_ratio)
                
                if num_to_mask > 0 and num_to_mask < num_timesteps:
                    # Select random positions to mask (only within real data)
                    valid_indices = torch.nonzero(mask == 1).squeeze()
                    masked_indices = torch.randperm(len(valid_indices))[:num_to_mask]
                    mask_positions = valid_indices[masked_indices]
                    
                    # Zero out masked positions
                    x[:, mask_positions] = 0
            
            # Add noise (only on real data, not padding)
            if self.config.add_noise:
                noise = torch.randn_like(x) * self.config.noise_std
                noise_scaled = noise * self.feature_std.squeeze()[:, None]
                
                if self.masks is not None:
                    mask = self.masks[idx].unsqueeze(0)  # (1, length)
                    noise_scaled = noise_scaled * mask
                
                x = x + noise_scaled
        
        return x, y

class Trainer:
    """Model trainer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Using device: CPU")
        
        self._create_directories()
        # Don't save config.json per model (redundant - final_results.json has full config)
        
        print("\n" + "="*60)
        print("PREPROCESSING")
        print("="*60)
        
        self.preprocessor = DataPreprocessor(config)
        X_train, y_train, stats_train = self.preprocessor.preprocess_train(config.train_path)
        self.stats_train = stats_train
        
        X_val, y_val = self.preprocessor.get_validation_chunks()
        
        # Get masks from preprocessor
        masks_train = getattr(self.preprocessor, 'masks_train', None)
        masks_val = getattr(self.preprocessor, 'masks_val', None)
        
        train_dataset = ChunkDataset(X_train, y_train, masks=masks_train, config=config, is_training=True)
        train_eval_dataset = ChunkDataset(X_train, y_train, masks=masks_train, config=config, is_training=False)
        val_dataset = ChunkDataset(X_val, y_val, masks=masks_val, config=config, is_training=False)
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.train_eval_loader = DataLoader(train_eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        
        print("\n" + "="*60)
        print("MODEL")
        print("="*60)
        
        self.model = create_model(config).to(self.device)
        num_params = count_parameters(self.model)
        print(f"Model: {num_params:,} parameters")
        print(f"Receptive field: {config.receptive_field}")
        
        # Calculate class weights
        class_names = config.get_class_names()
        
        # Use fixed class distribution if configured
        if getattr(config, 'use_fixed_class_weights', False):
            fixed_dist = getattr(config, 'fixed_class_distribution', {})
            class_counts = np.array([fixed_dist[name] for name in class_names])
        else:
            # Use original stats (before augmentation) for class weights
            # This prevents distortion from augmented data distribution
            original_stats = getattr(self.preprocessor, 'original_stats', stats_train)
            class_counts = np.array([original_stats[name]['count'] for name in class_names])
        
        total = class_counts.sum()
        class_weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
        class_weights = class_weights / class_weights.sum()
        
        print(f"\nClass weights:")
        for name, weight in zip(class_names, class_weights):
            print(f"  {name}: {weight:.4f}")
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.y_val = torch.LongTensor(y_val)
        
        # Determine early stopping metric
        self.use_val_acc = getattr(config, 'early_stopping_metric', 'val_loss') == 'val_acc'
    
    def _create_directories(self):
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        if self.config.logs_dir is not None:
            os.makedirs(self.config.logs_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Train")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(x)  # (batch, num_classes)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def train(self):
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model based on chosen metric
            if self.use_val_acc:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model(is_best=True)
                    print(f"  [OK] New best model saved! (val_acc: {val_acc:.4f})")
                else:
                    self.patience_counter += 1
            else:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model(is_best=True)
                    print("  [OK] New best model saved!")
                else:
                    self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            print()
        
        # Don't save final model or history (only best model is saved)
        print(f"\nTraining completed! Best model already saved.")
    
    def _evaluate_loader(self, loader):
        self.model.eval()
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                all_probabilities.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        if not all_probabilities:
            return np.empty((0, self.config.num_classes), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
        probabilities = np.concatenate(all_probabilities, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return probabilities, labels
    
    def get_validation_predictions(self):
        """Return probabilities and ground truth labels for the validation loader."""
        return self._evaluate_loader(self.val_loader)
    
    def get_train_predictions(self):
        """Return probabilities and ground truth labels for the training set without augmentation."""
        return self._evaluate_loader(self.train_eval_loader)
    
    def save_model(self, is_best=True):
        if is_best:
            # Only save best model (rename to just 'best_model.pth' for simplicity)
            path = os.path.join(self.config.model_dir, "best_model.pth")
            torch.save(self.model.state_dict(), path)
        # Don't save final model (only best is needed)
    
    def save_history(self):
        # Don't save history to reduce clutter
        # History is only used during training for early stopping
        pass

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()

