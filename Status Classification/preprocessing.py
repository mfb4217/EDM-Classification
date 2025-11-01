"""
Preprocessing for status classification
Load complete series and label by folder (Normal, NPT, OD)
"""
import numpy as np
import pandas as pd
import os
import glob
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from config import Config

class DataPreprocessor:
    """Data preprocessor for status classification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.max_length = getattr(config, 'max_series_length', 10000)
        
    def load_raw_data(self, data_path: str) -> List[Tuple[np.ndarray, int]]:
        """
        Load raw data from CSV files
        Returns: List of tuples (features, status_label)
        features: (length, 2) or (length, 4) - [Voltage, Z] or [Voltage, Z, dV/dt, dZ/dt]
        status_label: 0 (Normal), 1 (NPT), 2 (OD)
        """
        csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
        data = []
        
        print(f"Loading {len(csv_files)} files from {data_path}...")
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # Extract features (Voltage, Z)
                features = df[['Voltage', 'Z']].values.astype(np.float32)
                
                # Get status from folder name
                path_parts = file_path.split(os.sep)
                status_label = None
                for status, label in self.config.status_mapping.items():
                    if status in path_parts:
                        status_label = label
                        break
                
                if status_label is None:
                    # Skip non-target categories
                    continue
                
                # Add derivatives if configured
                if self.config.include_derivatives:
                    dV = np.gradient(features[:, 0])
                    dZ = np.gradient(features[:, 1])
                    features = np.column_stack([features, dV, dZ]).astype(np.float32)
                
                data.append((features, status_label))
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(data)} files")
        return data
    
    def pad_or_truncate(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or truncate series to fixed length
        Returns: (padded_features, mask) where mask indicates real data (1) vs padding (0)
        """
        current_length = len(features)
        
        if current_length >= self.max_length:
            # Truncate from the end
            padded = features[:self.max_length]
            mask = np.ones(self.max_length, dtype=np.float32)
        else:
            # Pad with zeros at the BEGINNING
            padding_length = self.max_length - current_length
            padded = np.pad(features, ((padding_length, 0), (0, 0)), mode='constant', constant_values=0)
            mask = np.concatenate([np.zeros(padding_length, dtype=np.float32),
                                   np.ones(current_length, dtype=np.float32)])
        
        return padded, mask
    
    def preprocess_train(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Preprocess training data"""
        # Load data
        data = self.load_raw_data(data_path)
        
        # Split series BEFORE creating chunks to avoid data leakage
        from sklearn.model_selection import train_test_split
        series_indices = list(range(len(data)))
        train_indices, val_indices = train_test_split(
            series_indices,
            test_size=0.1,
            random_state=self.config.seed
        )
        
        train_series = [data[i] for i in train_indices]
        val_series = [data[i] for i in val_indices]
        
        # Store validation series
        self._val_series = val_series
        
        print(f"\nSeries split: {len(train_series)} train, {len(val_series)} validation")
        
        # Pad/truncate series to fixed length
        print(f"Padding/truncating series to length {self.max_length}...")
        all_padded_features = []
        all_masks = []
        all_labels = []
        
        for features, status_label in train_series:
            padded, mask = self.pad_or_truncate(features)
            all_padded_features.append(padded)
            all_masks.append(mask)
            all_labels.append(status_label)
        
        # Convert to numpy arrays
        X = np.array(all_padded_features)  # (n_series, max_length, num_channels)
        masks = np.array(all_masks)        # (n_series, max_length)
        y = np.array(all_labels)           # (n_series,)
        
        # Store masks for later use
        self.masks_train = masks
        
        # Standardize
        if self.config.normalize:
            print("Standardizing features...")
            num_channels = X.shape[2]
            X_reshaped = X.reshape(-1, num_channels)
            self.scaler.fit(X_reshaped)
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(X.shape).astype(np.float32)
        
        # Statistics
        stats = self._compute_stats(y, "Train")
        
        print(f"\nTraining dataset:")
        print(f"  - Number of series: {len(X)}")
        print(f"  - Shape X: {X.shape}")
        print(f"  - Shape y: {y.shape}")
        
        return X, y, stats
    
    def get_validation_chunks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation series (padded)"""
        if not hasattr(self, '_val_series'):
            raise ValueError("Must call preprocess_train() first")
        
        all_padded_features = []
        all_masks = []
        all_labels = []
        
        for features, status_label in self._val_series:
            padded, mask = self.pad_or_truncate(features)
            all_padded_features.append(padded)
            all_masks.append(mask)
            all_labels.append(status_label)
        
        X = np.array(all_padded_features)
        y = np.array(all_labels)
        masks = np.array(all_masks)
        self.masks_val = masks
        
        # Transform with fitted scaler
        if self.config.normalize:
            num_channels = X.shape[2]
            X_reshaped = X.reshape(-1, num_channels)
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(X.shape).astype(np.float32)
        
        stats = self._compute_stats(y, "Validation")
        
        print(f"\nValidation dataset:")
        print(f"  - Number of series: {len(X)}")
        print(f"  - Shape X: {X.shape}")
        print(f"  - Shape y: {y.shape}")
        
        return X, y
    
    def preprocess_test(self, test_path: str, variable_length: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess test data
        
        Args:
            variable_length: If True, return list of variable-length series. If False, pad/truncate to fixed length.
        """
        data = self.load_raw_data(test_path)
        
        if variable_length:
            # Return variable-length series (no padding/truncation)
            all_features = []
            all_labels = []
            
            for features, status_label in data:
                # Normalize only
                if self.config.normalize:
                    features_scaled = self.scaler.transform(features)
                else:
                    features_scaled = features.astype(np.float32)
                
                all_features.append(features_scaled)
                all_labels.append(status_label)
            
            y = np.array(all_labels)
            stats = self._compute_stats(y, "Test")
            
            print(f"\nTest dataset (variable length):")
            print(f"  - Number of series: {len(all_features)}")
            print(f"  - Series lengths - Min: {min(len(f) for f in all_features)}, "
                  f"Max: {max(len(f) for f in all_features)}, "
                  f"Avg: {sum(len(f) for f in all_features) / len(all_features):.1f}")
            
            return all_features, y
        else:
            # Pad/truncate to fixed length (for training)
            all_padded_features = []
            all_masks = []
            all_labels = []
            
            for features, status_label in data:
                padded, mask = self.pad_or_truncate(features)
                all_padded_features.append(padded)
                all_masks.append(mask)
                all_labels.append(status_label)
            
            X = np.array(all_padded_features)
            y = np.array(all_labels)
            
            # Transform with fitted scaler
            if self.config.normalize:
                num_channels = X.shape[2]
                X_reshaped = X.reshape(-1, num_channels)
                X_scaled = self.scaler.transform(X_reshaped)
                X = X_scaled.reshape(X.shape).astype(np.float32)
            
            stats = self._compute_stats(y, "Test")
            
            print(f"\nTest dataset:")
            print(f"  - Number of series: {len(X)}")
            print(f"  - Shape X: {X.shape}")
            print(f"  - Shape y: {y.shape}")
            
            return X, y
    
    def _compute_stats(self, labels: np.ndarray, split: str) -> Dict:
        """Compute class distribution statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        
        total = len(labels)
        stats = {}
        
        print(f"\n{split} class distribution:")
        for label, count in zip(unique, counts):
            class_name = self.config.get_class_names()[label]
            percentage = 100 * count / total
            stats[class_name] = {'count': int(count), 'percentage': float(percentage)}
            print(f"  - {class_name}: {count} ({percentage:.1f}%)")
        
        return stats

