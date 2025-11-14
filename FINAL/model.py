"""
State-of-the-art lightweight model for status classification
Uses residual dilated convolutions with efficient design for limited data
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

class EfficientTCN(nn.Module):
    """
    Efficient TCN for series-level status classification
    Based on modern architectures optimized for small datasets
    
    Key features:
    - Depthwise separable convolutions (MobileNet style)
    - Residual connections (ResNet style)  
    - Dilated convolutions (TCN style)
    - GELU activations (BERT style)
    - Dense layers before final prediction
    """
    
    def __init__(self, config, debug=False):
        super().__init__()
        self.config = config
        self.debug = debug
        
        # Efficient backbone
        self.features = nn.ModuleList()
        in_channels = config.num_input_channels
        
        # Build residual dilated blocks
        strides = getattr(config, 'strides', [1] * len(config.channels))  # Default stride=1 if not provided
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
        
        # Dense layers (SOTA pattern)
        feature_dim = in_channels * (2 if config.use_max_pooling else 1)  # avg + max pooling
        dense_hidden = max(int(feature_dim * config.dense_hidden_ratio), config.dense_hidden_min)
        
        # Select activation for classifier
        if config.activation.lower() == "gelu":
            activation_fn = nn.GELU()
        elif config.activation.lower() == "swish":
            activation_fn = nn.SiLU()  # SiLU is Swish
        elif config.activation.lower() == "elu":
            activation_fn = nn.ELU()
        else:
            activation_fn = nn.ReLU()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, dense_hidden),
            activation_fn,
            nn.Dropout(config.dropout),
            nn.Linear(dense_hidden, config.num_classes)
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
    
    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Feature extraction
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.debug:
                print(f"After layer {i}: {x.shape}")
        
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
        
        # Classification
        logits = self.classifier(features)
        
        if self.debug:
            print(f"Output: {logits.shape}\n")
        
        return logits
    
    def predict_proba(self, x):
        """Predict probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """Predict classes"""
        probs = self.predict_proba(x)
        predictions = torch.argmax(probs, dim=1)
        return predictions

def create_model(config, debug=False):
    model = EfficientTCN(config, debug=debug)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    from config import Config
    
    config = Config()
    model = create_model(config, debug=True)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Receptive field: {config.receptive_field}")
    
    # Test
    batch_size = 4
    length = 600
    x = torch.randn(batch_size, 4, length)  # 4 channels with derivatives
    
    print(f"\nTest:")
    print(f"Input: {x.shape}")
    out = model(x)
    print(f"Output: {out.shape}")
    
    probs = model.predict_proba(x)
    preds = model.predict(x)
    print(f"Predictions: {preds}")

