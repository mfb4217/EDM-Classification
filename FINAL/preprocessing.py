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
        self.exclude_files_all = set()
        self.exclude_files_by_split = {}
        self._option2_filenames = None
        self._load_exclusion_list()
    
    def _load_option2_filenames(self) -> set:
        """Load filenames (base names) from Option 2 Train to avoid in validation split."""
        if self._option2_filenames is not None:
            return self._option2_filenames
        
        option2_path = getattr(self.config, 'option2_train_path', None)
        filenames = set()
        
        if option2_path and os.path.exists(option2_path):
            csv_files = glob.glob(os.path.join(option2_path, "**", "*.csv"), recursive=True)
            filenames = {os.path.basename(path) for path in csv_files}
            if filenames:
                print(f"Loaded {len(filenames)} filenames from Option 2 Train for validation exclusion.")
        self._option2_filenames = filenames
        return filenames

    def _load_exclusion_list(self):
        """Load list of files to exclude from processing, if provided."""
        csv_path = getattr(self.config, 'exclude_files_csv', None)
        if not csv_path:
            return

        if not os.path.exists(csv_path):
            print(f"Warning: exclusion CSV not found at {csv_path}. Continuing without exclusions.")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: could not read exclusion CSV ({csv_path}): {exc}. Continuing without exclusions.")
            return

        if 'filename' not in df.columns:
            print(f"Warning: exclusion CSV ({csv_path}) missing 'filename' column. Continuing without exclusions.")
            return

        filenames = df['filename'].astype(str).str.strip()
        self.exclude_files_all = set(filenames)

        if 'split' in df.columns:
            split_series = df['split'].astype(str).str.lower().str.strip()
            for split_name, subset in df.groupby(split_series):
                subset_filenames = subset['filename'].astype(str).str.strip()
                self.exclude_files_by_split[split_name] = set(subset_filenames)

        if self.exclude_files_all:
            print(f"Loaded {len(self.exclude_files_all)} files to exclude from {csv_path}.")

    def _infer_split(self, data_path: str) -> str:
        """Infer split name from the data path (e.g., train/test)."""
        base = os.path.basename(os.path.normpath(data_path)).lower()
        if base in {'train', 'test', 'val', 'validation'}:
            return 'train' if base == 'validation' else base

        for candidate in ['train', 'test', 'validation', 'val']:
            if candidate in base:
                return 'train' if candidate in {'validation', 'val'} else candidate

        return ''

    def _should_exclude(self, filename: str, split_name: str) -> bool:
        """Determine whether a given filename should be excluded for the split."""
        if not self.exclude_files_all:
            return False

        filename = filename.strip()
        split_key = split_name.lower() if split_name else ''

        if split_key and split_key in self.exclude_files_by_split:
            return filename in self.exclude_files_by_split[split_key]

        return filename in self.exclude_files_all
        
    def load_raw_data(self, data_path: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Load raw data from CSV files
        Returns: List of tuples (features, status_label, relative_path)
        features: (length, 2) or (length, 4) - [Voltage, Z] or [Voltage, Z, dV/dt, dZ/dt]
        status_label: 0 (Normal), 1 (NPT), 2 (OD)
        """
        csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
        data = []

        split_name = self._infer_split(data_path)
        excluded_count = 0
        
        print(f"Loading {len(csv_files)} files from {data_path}...")
        
        for file_path in csv_files:
            try:
                filename = os.path.basename(file_path)
                if self._should_exclude(filename, split_name):
                    excluded_count += 1
                    continue

                df = pd.read_csv(file_path)
                
                # Extract features (Voltage, Z)
                features = df[['Voltage', 'Z']].values.astype(np.float32)
                relative_path = os.path.relpath(file_path, data_path)
                
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
                
                data.append((features, status_label, relative_path))
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(data)} files")
        if excluded_count:
            target_split = split_name if split_name else "dataset"
            print(f"Excluded {excluded_count} files from {target_split} based on removal list.")
        return data
    
    def _split_series(self, data: List[Tuple[np.ndarray, int, str]]) -> Tuple[List[Tuple[np.ndarray, int, str]], List[Tuple[np.ndarray, int, str]]]:
        """Split series into train and validation sets."""
        predefined_split = getattr(self.config, 'predefined_split', None)
        validation_ratio = getattr(self.config, 'validation_split', 0.1)
        
        if predefined_split:
            train_keys = set(predefined_split.get('train', []))
            val_keys = set(predefined_split.get('val', []))
            train_series = []
            val_series = []
            
            for item in data:
                rel_path = item[2]
                if rel_path in val_keys:
                    val_series.append(item)
                else:
                    # Default to train if not explicitly in validation list
                    train_series.append(item)
            
            missing_val = len(val_keys) - len(val_series)
            if missing_val > 0:
                print(f"Warning: {missing_val} validation files listed in predefined split were not found.")
        else:
            option2_filenames = self._load_option2_filenames()
            series_indices = list(range(len(data)))
            
            if option2_filenames:
                non_overlap_indices = [
                    idx for idx, series in enumerate(data)
                    if os.path.basename(series[2]) not in option2_filenames
                ]
                overlap_indices = [idx for idx in series_indices if idx not in non_overlap_indices]
                
                desired_val = max(1, int(round(len(data) * validation_ratio)))
                if len(non_overlap_indices) < desired_val:
                    desired_val = len(non_overlap_indices)
                
                if desired_val > 0:
                    rng = np.random.RandomState(self.config.seed)
                    val_indices = set(rng.choice(non_overlap_indices, size=desired_val, replace=False).tolist())
                    train_indices = [idx for idx in series_indices if idx not in val_indices]
                else:
                    # Fallback to original behaviour if no suitable non-overlap samples
                    from sklearn.model_selection import train_test_split
                    train_indices, val_indices = train_test_split(
                        series_indices,
                        test_size=validation_ratio,
                        random_state=self.config.seed
                    )
                    val_indices = set(val_indices)
            else:
                from sklearn.model_selection import train_test_split
                train_indices, val_indices = train_test_split(
                    series_indices,
                    test_size=validation_ratio,
                    random_state=self.config.seed
                )
                val_indices = set(val_indices)
                train_indices = list(train_indices)
            
            train_series = [data[i] for i in series_indices if i not in val_indices]
            val_series = [data[i] for i in series_indices if i in val_indices]
            
            split_dict = {
                'train': [series[2] for series in train_series],
                'val': [series[2] for series in val_series]
            }
            setattr(self.config, 'predefined_split', split_dict)
        
        self._val_series = val_series
        return train_series, val_series
    
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
        
        train_series, val_series = self._split_series(data)
        
        # Calculate stats from original data (before augmentation) for class weights
        original_labels = [series[1] for series in train_series]
        original_y = np.array(original_labels)
        original_stats = self._compute_stats(original_y, "Train (original, before augmentation)")
        
        # Store original stats for class weight calculation
        self.original_stats = original_stats
        
        augmented_path = getattr(self.config, 'augmented_data_path', None)
        
        if augmented_path:
            if not os.path.exists(augmented_path):
                print(f"Warning: augmented data path {augmented_path} does not exist. Skipping augmented data.")
            else:
                augmented_series = self.load_raw_data(augmented_path)
                if augmented_series:
                    print(f"Adding {len(augmented_series)} augmented series to training set.")
                    train_series.extend(augmented_series)
                else:
                    print("Augmented data path provided but no series were loaded.")
        
        print(f"\nSeries split: {len(train_series)} train, {len(val_series)} validation")
        
        # Pad/truncate series to fixed length
        print(f"Padding/truncating series to length {self.max_length}...")
        all_padded_features = []
        all_masks = []
        all_labels = []
        
        for features, status_label, _ in train_series:
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
        
        for features, status_label, _ in self._val_series:
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
            
            for features, status_label, _ in data:
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
            
            for features, status_label, _ in data:
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

