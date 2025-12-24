"""
Contrastive loss functions for time series self-supervised learning
Based on TS2Vec hierarchical contrastive learning approach
Includes SoftCLT (Soft Contrastive Learning for Time Series) implementation
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, temporal_stride=1):
    """
    Hierarchical contrastive loss with optional temporal striding for memory efficiency.
    
    Args:
        z1, z2: Representations of shape (B, T, C)
        alpha: Weight for instance contrastive loss (default: 0.5)
        temporal_unit: Minimum unit to perform temporal contrast (default: 0)
        temporal_stride: Sample every Nth timestep for temporal contrast (default: 1)
                        Higher values reduce memory usage at the cost of temporal resolution.
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, temporal_stride=temporal_stride)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    """
    Instance contrastive loss - compares representations across different time series instances.
    
    Args:
        z1, z2: Representations of shape (B, T, C)
        
    Returns:
        loss: Scalar contrastive loss
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2, temporal_stride=1):
    """
    Temporal contrastive loss with optional temporal striding for memory efficiency.
    Compares representations across different timestamps within the same instance.
    
    Args:
        z1, z2: Representations of shape (B, T, C)
        temporal_stride: Sample every Nth timestep (default: 1 = all timesteps)
                        Higher values reduce memory usage at the cost of temporal resolution.
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    # Sample timesteps with stride for memory efficiency
    if temporal_stride > 1:
        # Sample indices: [0, stride, 2*stride, ...]
        indices = torch.arange(0, T, temporal_stride, device=z1.device)
        z1_sampled = z1[:, indices, :]  # B x (T//stride) x C
        z2_sampled = z2[:, indices, :]  # B x (T//stride) x C
        T_sampled = z1_sampled.size(1)
    else:
        z1_sampled = z1
        z2_sampled = z2
        T_sampled = T
    
    # Concatenate sampled timesteps
    z = torch.cat([z1_sampled, z2_sampled], dim=1)  # B x 2T_sampled x C
    
    # Compute similarity matrix (now much smaller with striding!)
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T_sampled x 2T_sampled
    
    # Build logits (same as before)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T_sampled x (2T_sampled-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    # Extract loss for corresponding timesteps
    t = torch.arange(T_sampled, device=z1.device)
    loss = (logits[:, t, T_sampled + t - 1].mean() + logits[:, T_sampled + t, t].mean()) / 2
    
    return loss


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss for time series embeddings.
    
    Args:
        anchor: Anchor embeddings (B, C)
        positive: Positive embeddings (B, C) - same instance, different view
        negative: Negative embeddings (B, C) - different instance
        margin: Margin for triplet loss
        
    Returns:
        loss: Scalar triplet loss
    """
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = F.relu(distance_positive - distance_negative + margin)
    return loss.mean()


########################################################################################################
## SoftCLT: Soft Contrastive Learning for Time Series
## Based on: https://github.com/seunghan96/softclt
########################################################################################################

def dup_matrix(mat):
    """
    Duplicate matrix for soft contrastive loss.
    Creates matrices for left and right views.
    """
    mat0 = torch.tril(mat, diagonal=-1)[:, :-1]   
    mat0 += torch.triu(mat, diagonal=1)[:, 1:]
    mat1 = torch.cat([mat0, mat], dim=1)
    mat2 = torch.cat([mat, mat0], dim=1)
    return mat1, mat2


def timelag_sigmoid(T, sigma=1.0):
    """
    Generate temporal soft assignments using sigmoid function.
    Closer timestamps get higher soft assignment values.
    
    Args:
        T: Sequence length
        sigma: Temperature parameter (higher = sharper decay)
        
    Returns:
        matrix: (T, T) matrix with soft assignments based on timestamp distance
    """
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 + np.exp(dist * sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)  # Set very small values to 0
    return matrix


def timelag_gaussian(T, sigma=1.0):
    """
    Generate temporal soft assignments using Gaussian function.
    
    Args:
        T: Sequence length
        sigma: Standard deviation (higher = wider distribution)
        
    Returns:
        matrix: (T, T) matrix with soft assignments based on timestamp distance
    """
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = np.exp(-(dist**2) / (2 * sigma**2))
    matrix = np.where(matrix < 1e-6, 0, matrix)
    return matrix


def instance_contrastive_loss_soft(z1, z2, soft_labels_L, soft_labels_R):
    """
    Instance-wise soft contrastive loss.
    Uses soft assignments instead of hard 0/1 labels.
    
    Args:
        z1, z2: Representations of shape (B, T, C)
        soft_labels_L: Soft assignment matrix for left view (B, 2B-1)
        soft_labels_R: Soft assignment matrix for right view (B, 2B-1)
        
    Returns:
        loss: Scalar contrastive loss
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = torch.sum(logits[:, i] * soft_labels_L)
    loss += torch.sum(logits[:, B + i] * soft_labels_R)
    loss /= (2 * B * T)
    return loss


def temporal_contrastive_loss_soft(z1, z2, timelag_L, timelag_R):
    """
    Temporal soft contrastive loss.
    Uses soft assignments based on timestamp distance.
    
    Args:
        z1, z2: Representations of shape (B, T, C)
        timelag_L: Soft assignment matrix for left view (T, 2T-1)
        timelag_R: Soft assignment matrix for right view (T, 2T-1)
        
    Returns:
        loss: Scalar contrastive loss
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = torch.sum(logits[:, t] * timelag_L)
    loss += torch.sum(logits[:, T + t] * timelag_R)
    loss /= (2 * B * T)
    return loss


def hierarchical_contrastive_loss_soft(z1, z2, soft_labels=None, tau_temp=2.0, lambda_=0.5, 
                                       temporal_unit=0, soft_temporal=False, soft_instance=False, 
                                       temporal_hierarchy=True, temporal_stride=1):
    """
    Hierarchical soft contrastive loss combining instance and temporal contrastive learning.
    
    Args:
        z1, z2: Representations of shape (B, T, C)
        soft_labels: Soft assignment matrix for instances (B, B) - similarity between time series
                    If None, uses hard instance contrastive loss
        tau_temp: Temperature parameter for temporal soft assignments (higher = sharper decay)
        lambda_: Weight for instance contrastive loss (1-lambda_ for temporal)
        temporal_unit: Minimum unit to perform temporal contrast (default: 0)
        soft_temporal: Whether to use soft temporal contrastive loss
        soft_instance: Whether to use soft instance contrastive loss
        temporal_hierarchy: Whether to scale tau_temp with hierarchy level
        temporal_stride: Sample every Nth timestep for temporal contrast (default: 1)
        
    Returns:
        loss: Scalar contrastive loss
    """
    if soft_labels is not None:
        soft_labels = torch.tensor(soft_labels, device=z1.device, dtype=z1.dtype)
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        # Instance contrastive loss
        if lambda_ != 0:
            if soft_instance and soft_labels is not None:
                loss += lambda_ * instance_contrastive_loss_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * instance_contrastive_loss(z1, z2)
        
        # Temporal contrastive loss
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    # Sample timesteps if stride > 1
                    T_current = z1.size(1)
                    if temporal_stride > 1:
                        indices = torch.arange(0, T_current, temporal_stride, device=z1.device)
                        z1_temp = z1[:, indices, :]
                        z2_temp = z2[:, indices, :]
                        T_sampled = z1_temp.size(1)
                    else:
                        z1_temp = z1
                        z2_temp = z2
                        T_sampled = T_current
                    
                    # Generate timelag matrix
                    if temporal_hierarchy:
                        tau = tau_temp * (2 ** d)
                    else:
                        tau = tau_temp
                    
                    timelag = timelag_sigmoid(T_sampled, sigma=tau)
                    timelag = torch.tensor(timelag, device=z1.device, dtype=z1.dtype)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    
                    loss += (1 - lambda_) * temporal_contrastive_loss_soft(z1_temp, z2_temp, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temporal_contrastive_loss(z1, z2, temporal_stride=temporal_stride)
        
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance and soft_labels is not None:
                loss += lambda_ * instance_contrastive_loss_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * instance_contrastive_loss(z1, z2)
        d += 1
    
    return loss / d if d > 0 else loss


def compute_soft_labels_from_data(X, dist_type='euclidean', min_val=0, max_val=1):
    """
    Compute soft assignment matrix from time series data using distance metrics.
    This is used for instance-wise soft contrastive learning.
    
    Args:
        X: Time series data of shape (N, T, C) or (N, T)
        dist_type: Distance metric ('euclidean', 'dtw', 'cosine')
        min_val: Minimum value for normalization (default: 0)
        max_val: Maximum value for normalization (default: 1)
        
    Returns:
        soft_labels: (N, N) similarity matrix where values range from min_val to max_val
                    Higher values indicate more similar time series
    """
    N = X.shape[0]
    
    # Flatten multivariate time series to 2D if needed
    if len(X.shape) == 3:
        X_flat = X.reshape(N, -1)  # (N, T*C)
    else:
        X_flat = X  # (N, T)
    
    # Compute distance matrix
    if dist_type == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist_mat = euclidean_distances(X_flat)
    elif dist_type == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        dist_mat = -cosine_similarity(X_flat)  # Negative because cosine_similarity returns similarity
    elif dist_type == 'dtw':
        try:
            from tslearn.metrics import dtw
            dist_mat = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    if len(X.shape) == 3:
                        dist = dtw(X[i], X[j])
                    else:
                        dist = dtw(X[i].reshape(-1, 1), X[j].reshape(-1, 1))
                    dist_mat[i, j] = dist
                    dist_mat[j, i] = dist
        except ImportError:
            print("Warning: tslearn not available, falling back to euclidean distance")
            from sklearn.metrics.pairwise import euclidean_distances
            dist_mat = euclidean_distances(X_flat)
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")
    
    # Normalize distance matrix
    # Set diagonal to minimum off-diagonal value
    diag_indices = np.diag_indices(N)
    mask = np.ones(dist_mat.shape, dtype=bool)
    mask[diag_indices] = False
    temp = dist_mat[mask].reshape(N, N-1)
    dist_mat[diag_indices] = temp.min()
    
    # Normalize to [min_val, max_val]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    dist_mat = scaler.fit_transform(dist_mat)
    
    # Convert distance to similarity (1 - normalized_distance)
    soft_labels = 1 - dist_mat
    
    return soft_labels

