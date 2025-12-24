"""
Custom Foundational Model for Time Series Embeddings
Based on FINAL model architecture but without classification layers
Uses residual dilated convolutions for feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedBlock(nn.Module):
    """Residual dilated convolution block with configurable design"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, dropout=0.1, 
                 use_depthwise=True, use_residual=True, activation="gelu"):
        super().__init__()
        # Adjust padding for stride
        if stride > 1:
            padding = ((kernel_size - 1) * dilation) // 2
        else:
            padding = ((kernel_size - 1) * dilation + 1) // 2
        
        # Use depthwise separable conv for efficiency or standard conv
        if use_depthwise:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                         dilation=dilation, padding=padding, padding_mode='replicate',
                         groups=in_channels),  # Depthwise
                nn.Conv1d(in_channels, out_channels, 1)  # Pointwise
            )
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                 dilation=dilation, padding=padding, padding_mode='replicate')
        
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Select activation
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()  # SiLU is Swish (x * sigmoid(x))
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv with stride if needed)
        if use_residual:
            if in_channels != out_channels or stride > 1:
                self.residual = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            else:
                self.residual = nn.Identity()
            self.use_residual = True
        else:
            self.residual = None
            self.use_residual = False
    
    def forward(self, x):
        residual = self.residual(x) if self.use_residual else 0
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual if self.use_residual else out
        return out


class CustomFoundationalEncoder(nn.Module):
    """
    Custom foundational encoder based on FINAL model architecture
    Uses residual dilated convolutions for feature extraction
    Outputs embeddings instead of classification logits
    """
    
    def __init__(self, config, embedding_dim=256, debug=False):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.debug = debug
        
        # Feature extraction backbone (same as FINAL model)
        self.features = nn.ModuleList()
        in_channels = config.num_input_channels
        
        # Build residual dilated blocks
        strides = getattr(config, 'strides', [1] * len(config.channels))
        for out_channels, dilation, stride in zip(config.channels, config.dilations, strides):
            self.features.append(
                ResidualDilatedBlock(
                    in_channels, out_channels, config.kernel_size, dilation, stride=stride,
                    dropout=config.dropout,
                    use_depthwise=config.use_depthwise_separable,
                    use_residual=config.use_residual,
                    activation=config.activation
                )
            )
            in_channels = out_channels
        
        # Global pooling (average + max for better features)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        if config.use_max_pooling:
            self.pool_max = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool_max = None
        
        # Projection layer to embedding dimension (replaces classifier)
        feature_dim = in_channels * (2 if config.use_max_pooling else 1)  # avg + max pooling
        
        # Select activation for projection
        if config.activation.lower() == "gelu":
            activation_fn = nn.GELU()
        elif config.activation.lower() == "swish":
            activation_fn = nn.SiLU()  # SiLU is Swish
        elif config.activation.lower() == "elu":
            activation_fn = nn.ELU()
        else:
            activation_fn = nn.ReLU()
        
        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            activation_fn,
            nn.Dropout(config.dropout),
            nn.Linear(embedding_dim, embedding_dim)  # Final embedding layer
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for convolutional layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_temporal=False):
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, length)
            return_temporal: If True, also return temporal features before pooling
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            temporal_features: (optional) Tensor of shape (batch_size, channels, temporal_length)
        """
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Feature extraction
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.debug:
                print(f"After layer {i}: {x.shape}")
        
        # Store temporal features before pooling (for SoftCLT)
        temporal_features = x if return_temporal else None  # (batch, channels, temporal_length)
        
        # Global pooling (avg + max)
        avg_pool = self.pool_avg(x).squeeze(-1)  # (batch, channels)
        
        if self.pool_max is not None:
            max_pool = self.pool_max(x).squeeze(-1)  # (batch, channels)
            # Concatenate
            features = torch.cat([avg_pool, max_pool], dim=1)  # (batch, channels*2)
        else:
            features = avg_pool  # (batch, channels)
        
        if self.debug:
            print(f"Pooled features: {features.shape}")
        
        # Project to embedding dimension
        embeddings = self.projection(features)  # (batch, embedding_dim)
        
        if self.debug:
            print(f"Embeddings: {embeddings.shape}\n")
        
        if return_temporal:
            return embeddings, temporal_features
        return embeddings
    
    def encode(self, x):
        """Alias for forward method (for compatibility with TS2Vec interface)"""
        return self.forward(x)


def create_custom_foundational_model(config, embedding_dim=256, debug=False):
    """
    Create custom foundational encoder model
    
    Args:
        config: Configuration object with model architecture parameters
        embedding_dim: Dimension of output embeddings (default: 256)
        debug: Whether to print debug information
        
    Returns:
        CustomFoundationalEncoder model
    """
    model = CustomFoundationalEncoder(config, embedding_dim=embedding_dim, debug=debug)
    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    from FINAL.config import Config, load_config_from_json
    import os
    
    # Create a minimal config for testing
    class TestConfig:
        num_input_channels = 2
        channels = [64, 128, 256, 512, 512]
        dilations = [1, 2, 4, 8, 16]
        strides = [2, 2, 2, 2, 1]
        kernel_size = 7
        dropout = 0.3
        use_depthwise_separable = False
        use_residual = True
        activation = "swish"
        use_max_pooling = True
    
    config = TestConfig()
    model = create_custom_foundational_model(config, embedding_dim=256, debug=True)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    length = 5000
    x = torch.randn(batch_size, 2, length)  # 2 channels (Voltage, Z)
    
    print(f"\nTest:")
    print(f"Input: {x.shape}")
    embeddings = model(x)
    print(f"Output embeddings: {embeddings.shape}")

