# SoftCLT (Soft Contrastive Learning for Time Series) Usage Guide

## Overview

SoftCLT is a plug-and-play method that improves time series contrastive learning by using **soft assignments** (0.0 to 1.0) instead of hard binary assignments (0 or 1). This preserves the inherent correlations between similar time series instances and adjacent timestamps.

## Key Concepts

### Hard vs Soft Assignments

**Hard Contrastive Learning (Traditional):**
- Positive pairs: assignment = 1.0 (must be similar)
- Negative pairs: assignment = 0.0 (must be different)
- Problem: Treats all negatives equally, ignoring similarity structure

**Soft Contrastive Learning (SoftCLT):**
- Similar instances: assignment = 0.7-0.9 (soft positive)
- Dissimilar instances: assignment = 0.1-0.3 (soft negative)
- Adjacent timestamps: assignment = 0.8-0.9 (soft positive)
- Distant timestamps: assignment = 0.2-0.4 (soft negative)

## Implementation

The SoftCLT loss functions are implemented in `contrastive_losses.py`:

### Main Functions

1. **`hierarchical_contrastive_loss_soft()`** - Main function combining instance and temporal soft contrastive learning
2. **`instance_contrastive_loss_soft()`** - Instance-wise soft contrastive loss
3. **`temporal_contrastive_loss_soft()`** - Temporal soft contrastive loss
4. **`compute_soft_labels_from_data()`** - Utility to compute soft labels from time series data

### Helper Functions

- **`timelag_sigmoid()`** - Generate temporal soft assignments using sigmoid
- **`timelag_gaussian()`** - Generate temporal soft assignments using Gaussian
- **`dup_matrix()`** - Duplicate matrix for left/right views

## Usage Example

### Basic Usage

```python
from contrastive_losses import hierarchical_contrastive_loss_soft

# z1, z2: Representations of shape (B, T, C) - two views of the same data
# soft_labels: (B, B) similarity matrix (optional, for instance soft CL)

loss = hierarchical_contrastive_loss_soft(
    z1, z2,
    soft_labels=None,  # Set to None for hard instance CL
    tau_temp=2.0,       # Temperature for temporal soft assignments
    lambda_=0.5,        # Weight for instance CL (1-lambda_ for temporal)
    soft_temporal=True, # Use soft temporal CL
    soft_instance=False # Use hard instance CL (set True if soft_labels provided)
)
```

### With Instance Soft Labels

```python
from contrastive_losses import hierarchical_contrastive_loss_soft, compute_soft_labels_from_data
import numpy as np

# X: Training data of shape (N, T, C)
# Compute soft labels based on distance in data space
soft_labels = compute_soft_labels_from_data(
    X, 
    dist_type='euclidean',  # or 'dtw', 'cosine'
    min_val=0,
    max_val=1
)

# Use in training loop
for x1, x2 in dataloader:
    z1 = model(x1)  # (B, T, C)
    z2 = model(x2)  # (B, T, C)
    
    # Get soft labels for current batch
    batch_indices = ...  # Get indices for current batch
    batch_soft_labels = soft_labels[batch_indices][:, batch_indices]  # (B, B)
    
    loss = hierarchical_contrastive_loss_soft(
        z1, z2,
        soft_labels=batch_soft_labels,
        tau_temp=2.0,
        lambda_=0.5,
        soft_temporal=True,
        soft_instance=True
    )
```

### Temporal Soft CL Only

```python
# Use only temporal soft contrastive learning
loss = hierarchical_contrastive_loss_soft(
    z1, z2,
    soft_labels=None,
    tau_temp=2.0,
    lambda_=0.0,  # Disable instance CL
    soft_temporal=True,
    soft_instance=False
)
```

## Hyperparameters

### `tau_temp` (Temporal Temperature)
- **Lower values (0.5-1.0)**: Sharper decay, only very close timestamps get high soft assignment
- **Higher values (2.0-5.0)**: Gentler decay, more timestamps get moderate soft assignment
- **Default**: 2.0

### `lambda_` (Instance/Temporal Weight)
- **0.0**: Only temporal contrastive learning
- **0.5**: Equal weight (default)
- **1.0**: Only instance contrastive learning

### `soft_temporal` and `soft_instance`
- **True**: Use soft assignments
- **False**: Use hard assignments (traditional contrastive learning)

## Integration with Training Script

To use SoftCLT in `train_custom_foundational.py`:

1. **Import the function:**
```python
from contrastive_losses import hierarchical_contrastive_loss_soft, compute_soft_labels_from_data
```

2. **Compute soft labels (once, before training):**
```python
# Load all training data
train_data, _ = load_edm_unlabeled(...)

# Compute soft labels
soft_labels = compute_soft_labels_from_data(
    train_data,
    dist_type='euclidean',  # or 'dtw' for better accuracy (requires tslearn)
    min_val=0,
    max_val=1
)
```

3. **Use in training loop:**
```python
def train_epoch(model, train_loader, optimizer, device, config, soft_labels=None):
    for batch_idx, (x1, x2) in enumerate(train_loader):
        # Get batch indices
        batch_indices = ...  # Map batch to original data indices
        
        # Get soft labels for this batch
        if soft_labels is not None:
            batch_soft_labels = soft_labels[batch_indices][:, batch_indices]
        else:
            batch_soft_labels = None
        
        # Forward pass
        z1 = model(x1)  # (B, T, C)
        z2 = model(x2)  # (B, T, C)
        
        # Compute SoftCLT loss
        loss = hierarchical_contrastive_loss_soft(
            z1, z2,
            soft_labels=batch_soft_labels,
            tau_temp=config.get('tau_temp', 2.0),
            lambda_=config.get('lambda_', 0.5),
            soft_temporal=config.get('soft_temporal', True),
            soft_instance=(batch_soft_labels is not None),
            temporal_stride=config.get('temporal_stride', 1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Benefits

1. **Better Representation Quality**: Preserves similarity structure in learned embeddings
2. **Improved Downstream Performance**: Better classification, anomaly detection, etc.
3. **Plug-and-Play**: Can be used with any contrastive learning framework
4. **Flexible**: Can use soft assignments for instance, temporal, or both

## References

- Paper: "Soft Contrastive Learning for Time Series" (ICLR 2024)
- GitHub: https://github.com/seunghan96/softclt

