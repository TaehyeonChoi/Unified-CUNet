"""
Unified-CUNet: A Transformer-Based Framework for Surface-Aware 
Point Cloud Compression and Upsampling

This implementation follows the architecture described in the PLOS ONE paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    """
    Transformer block for processing point cloud features.
    
    Implements Equations (4)-(7) from the paper:
        X_a = A(F_l, F_l, F_l)           # Multi-head self-attention
        X_nl = N(F_l + D(X_a))           # Add & Norm
        X_f = Î¦(X_nl)                    # Feed-forward network
        F_{l+1} = N(X_nl + D(X_f))       # Add & Norm
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention (A)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
        
        # Layer normalization (N)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network (Î¦)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Dropout (D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input features [B, D, N] where B=batch, D=dimension, N=points
            
        Returns:
            Processed features [B, D, N]
        """
        batch_size, D, N = x.shape
        
        # Reshape for attention: [N, B, D] (sequence_length, batch, features)
        x_att = x.permute(2, 0, 1)  # [N, B, D]
        
        # Equation (4): Multi-head self-attention
        att_out, _ = self.attention(x_att, x_att, x_att)
        
        # Equation (5): Add & Norm (first residual connection)
        x_res = x_att + self.dropout(att_out)
        x_norm1 = self.norm1(x_res)  # [N, B, D]
        
        # Equation (6): Feed-forward network
        ffn_out = self.ffn(x_norm1)
        
        # Equation (7): Add & Norm (second residual connection)
        x_out = self.norm2(x_norm1 + self.dropout(ffn_out))
        
        # Reshape back: [B, D, N]
        return x_out.permute(1, 2, 0)


class MLPBlock(nn.Module):
    """
    Capacity-matched MLP replacement for TransformerBlock.

    Used in the transformer-vs-MLP ablation study (Reviewer 2, Comment 1).
    Mirrors the two-sub-block structure of TransformerBlock (attention +
    feed-forward) but replaces global self-attention with pointwise MLPs,
    deliberately removing the ability to model inter-point relationships.

    Parameter budget is matched to TransformerBlock at dim=64:
        TransformerBlock  ~50 k params  (attention ~17 k + FFN ~33 k)
        MLPBlock          ~66 k params  (mlp1 ~33 k + mlp2 ~33 k)
    The ~30 % difference is noted in the paper.

    Args:
        dim: Feature dimension (must match hidden_dim of the parent model)
        dropout: Dropout rate
    """
    def __init__(self, dim, dropout=0.1):
        super(MLPBlock, self).__init__()

        # Sub-block 1: replaces multi-head self-attention + residual + norm
        # Uses pointwise Conv1d so it stays permutation-invariant but has
        # NO cross-point communication — this is the key ablation variable.
        self.mlp1 = nn.Sequential(
            nn.Conv1d(dim, dim * 4, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 4, dim, 1)
        )
        self.norm1 = nn.LayerNorm(dim)

        # Sub-block 2: replaces feed-forward network + residual + norm
        self.mlp2 = nn.Sequential(
            nn.Conv1d(dim, dim * 4, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 4, dim, 1)
        )
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, D, N]
        Returns:
            [B, D, N]
        """
        # Sub-block 1
        out1 = self.mlp1(x)                                 # [B, D, N]
        # LayerNorm expects [..., D]; permute to [B, N, D]
        x = self.norm1(
            (x + self.dropout(out1)).permute(0, 2, 1)
        ).permute(0, 2, 1)                                  # back to [B, D, N]

        # Sub-block 2
        out2 = self.mlp2(x)
        x = self.norm2(
            (x + self.dropout(out2)).permute(0, 2, 1)
        ).permute(0, 2, 1)

        return x


class FarthestPointSampling(nn.Module):
    """
    Farthest Point Sampling (FPS) for selecting representative points.
    
    Implements Equation (9) from the paper:
        L_c = FPS(P_in, N_c)
    
    FPS iteratively selects points that are maximally distant from 
    the current selected set, ensuring uniform spatial coverage.
    """
    def __init__(self):
        super(FarthestPointSampling, self).__init__()

    def forward(self, xyz, npoint):
        """
        Args:
            xyz: Point cloud coordinates [B, N, 3]
            npoint: Number of points to sample
            
        Returns:
            centroids: Indices of sampled points [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        
        # Randomly select the first point
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, dim=-1)[1]
        
        return centroids


class FeatureExpansion(nn.Module):
    """
    Feature expansion module for upsampling.
    
    Implements the feature expansion described in the paper:
    - Pre-expansion transformation (Equation 12)
    - Learned offset generation (Equation 13)
    - Replication with adaptive offsets
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension  
        up_factor: Upsampling ratio (Î²)
    """
    def __init__(self, in_dim, out_dim, up_factor):
        super(FeatureExpansion, self).__init__()
        self.up_factor = up_factor
        self.out_dim = out_dim
        
        # Equation (12): Pre-expansion transformation
        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
        # Equation (13): Offset prediction network
        self.offset_conv = nn.Conv1d(out_dim, out_dim * up_factor, 1)

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, N]
            
        Returns:
            Expanded features [B, out_dim, N * up_factor]
        """
        batch_size, C, N = x.shape
        
        # Pre-expansion transformation
        x = self.pre_conv(x)  # [B, out_dim, N]
        
        # Generate learned offsets for each replicated point
        offsets = self.offset_conv(x)  # [B, out_dim * up_factor, N]
        offsets = offsets.view(batch_size, self.out_dim, self.up_factor, N)
        
        # Create base expansion via replication
        expanded = x.unsqueeze(2).repeat(1, 1, self.up_factor, 1)  # [B, out_dim, up_factor, N]
        
        # Add learned offsets (geometry-aware placement)
        expanded = expanded + offsets
        
        # Reshape to [B, out_dim, N * up_factor]
        expanded = expanded.permute(0, 1, 3, 2).contiguous()
        expanded = expanded.view(batch_size, self.out_dim, N * self.up_factor)
        
        return expanded


class UnifiedCUNet(nn.Module):
    """
    Unified-CUNet: Unified Compression-Upsampling Network
    
    A transformer-based framework that integrates point cloud compression 
    and upsampling within a unified architecture.
    
    Architecture Overview (Figure 1 in paper):
        Input â†’ Feature Extraction â†’ Transformer Blocks (Ã—4) â†’ 
        Compression Module â†’ Decompression Bridge â†’ 
        Feature Expansion â†’ Coordinate Reconstruction â†’ Output
    
    Args:
        point_dim: Input point dimension (default: 3 for xyz)
        hidden_dim: Hidden feature dimension (D in paper)
        cs_dim: Compressed feature dimension (D_c in paper)
        feature_dim: Feature dimension for upsampling
        num_transformer_blocks: Number of transformer blocks (L in paper)
        num_heads: Number of attention heads
        dropout: Dropout rate
        compression_ratio: Compression ratio Î± âˆˆ (0, 1)
        upsample_ratio: Upsampling ratio Î² > 1
    """
    def __init__(
        self,
        point_dim=3,
        hidden_dim=64,
        cs_dim=32,
        feature_dim=64,
        num_transformer_blocks=4,
        num_heads=8,
        dropout=0.1,
        compression_ratio=0.5,
        upsample_ratio=4.0
    ):
        super(UnifiedCUNet, self).__init__()
        
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.cs_dim = cs_dim
        self.feature_dim = feature_dim
        self.compression_ratio = compression_ratio
        self.upsample_ratio = upsample_ratio
        self.num_transformer_blocks = num_transformer_blocks
        
        # ============================================================
        # Feature Extraction (Equations 1-2)
        # E(P) = Ï†(Î·(C(Ï†(Î·(C(P))))))
        # ============================================================
        self.encoder = nn.Sequential(
            nn.Conv1d(point_dim, hidden_dim, 1),      # C: 1D convolution
            nn.BatchNorm1d(hidden_dim),               # Î·: Batch normalization
            nn.ReLU(),                                 # Ï†: ReLU activation
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # ============================================================
        # Transformer Blocks (Equation 3)
        # F_l = T_l(F_{l-1}), l = 1, 2, ..., L
        # ============================================================
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # ============================================================
        # Compression Module (Equations 8-10)
        # ============================================================
        # Feature dimension reduction (Equation 8)
        self.cs_conv = nn.Conv1d(hidden_dim, cs_dim, 1)
        
        # Point count reduction via FPS (Equation 9)
        self.fps = FarthestPointSampling()
        
        # ============================================================
        # Decompression Bridge (Equation 11)
        # F_d = Ï†(C(F^c))
        # ============================================================
        self.decompress_bridge = nn.Sequential(
            nn.Conv1d(cs_dim, hidden_dim, 1),
            nn.ReLU()
        )
        
        # ============================================================
        # Feature Expansion (Equations 12-13)
        # ============================================================
        self.feature_expansion = FeatureExpansion(
            hidden_dim, feature_dim, int(upsample_ratio)
        )
        
        # ============================================================
        # Coordinate Reconstruction (Equation 14)
        # P_out = D(F_e)
        # ============================================================
        self.coord_decoder = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim // 2, 1),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv1d(feature_dim // 2, point_dim, 1)
        )

    def forward(self, x, mode='full'):
        """
        Forward pass with flexible operational modes.
        
        Args:
            x: Input point cloud [B, 3, N]
            mode: Operation mode
                - 'compress': Compression only (P_in â†’ P_c)
                - 'upsample': Upsampling only (assumes compressed input)
                - 'full': End-to-end processing (P_in â†’ P_out)
                
        Returns:
            Processed point cloud tensor
        """
        batch_size = x.size(0)
        
        if mode in ['compress', 'full']:
            # ========================================
            # Feature Extraction (Equation 1)
            # F_o = E(P_in)
            # ========================================
            features = self.encoder(x)  # [B, hidden_dim, N]
            
            # ========================================
            # Transformer Processing (Equation 3)
            # F_l = T_l(F_{l-1})
            # ========================================
            for block in self.transformer_blocks:
                features = block(features)  # [B, hidden_dim, N]
            
            # ========================================
            # Compression Module
            # ========================================
            # Feature dimension reduction (Equation 8)
            cs_features = self.cs_conv(features)  # [B, cs_dim, N]
            
            # Point count reduction via FPS (Equations 9-10)
            if self.compression_ratio < 1.0:
                num_points = x.size(2)
                compressed_points = max(32, int(num_points * self.compression_ratio))
                
                # Get point coordinates for FPS
                xyz = x.transpose(1, 2)  # [B, N, 3]
                fps_indices = self.fps(xyz, compressed_points)  # [B, N_c]
                
                # Select compressed features (Equation 10)
                compressed_features = torch.stack([
                    cs_features[b, :, fps_indices[b]]
                    for b in range(batch_size)
                ])  # [B, cs_dim, N_c]
            else:
                compressed_features = cs_features
            
            if mode == 'compress':
                return compressed_features
        
        if mode in ['upsample', 'full']:
            if mode == 'upsample':
                compressed_features = x
            
            # ========================================
            # Decompression Bridge (Equation 11)
            # F_d = Ï†(C(F^c))
            # ========================================
            decompressed = self.decompress_bridge(compressed_features)
            
            # ========================================
            # Feature Expansion (Equations 12-13)
            # ========================================
            expanded_features = self.feature_expansion(decompressed)
            
            # ========================================
            # Coordinate Reconstruction (Equation 14)
            # P_out = D(F_e)
            # ========================================
            output_points = self.coord_decoder(expanded_features)
            
            return output_points
        
        raise ValueError(f"Unknown mode: {mode}. Use 'compress', 'upsample', or 'full'.")

    def get_config(self):
        """Return model configuration dictionary."""
        return {
            'name': 'Unified-CUNet',
            'point_dim': self.point_dim,
            'hidden_dim': self.hidden_dim,
            'cs_dim': self.cs_dim,
            'feature_dim': self.feature_dim,
            'compression_ratio': self.compression_ratio,
            'upsample_ratio': self.upsample_ratio,
            'num_transformer_blocks': self.num_transformer_blocks
        }


class DimensionHandler(nn.Module):
    """
    Wrapper that handles input/output dimension formats automatically.
    
    Point cloud tensors can be in two formats:
        - BCN: [Batch, Channels, NumPoints] - used internally
        - BNC: [Batch, NumPoints, Channels] - common in datasets
    
    This wrapper ensures consistent handling regardless of input format.
    """
    def __init__(self, model, input_format='BCN', name='Unified-CUNet'):
        super(DimensionHandler, self).__init__()
        self.model = model
        self.input_format = input_format
        self.name = name
        
    def forward(self, x, mode='full'):
        original_format = self._detect_format(x)
        
        # Convert to BCN if needed
        if original_format == 'BNC':
            x = x.permute(0, 2, 1).contiguous()
        
        # Forward through model
        output = self.model(x, mode=mode)
        
        # Convert back to original format if needed
        if original_format == 'BNC' and isinstance(output, torch.Tensor):
            output = output.permute(0, 2, 1).contiguous()
        
        return output
    
    def _detect_format(self, x):
        """Detect whether input is BCN or BNC format."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
        
        # Assume channel dimension is 3 (xyz coordinates)
        if x.size(1) == 3:
            return 'BCN'
        elif x.size(2) == 3:
            return 'BNC'
        else:
            # Default assumption based on typical sizes
            return 'BCN' if x.size(1) < x.size(2) else 'BNC'
    
    def get_config(self):
        return self.model.get_config()


def create_unified_cunet(
    compression_ratio=0.5,
    upsample_ratio=4.0,
    hidden_dim=64,
    num_transformer_blocks=4,
    with_dimension_handler=True
):
    """
    Factory function to create Unified-CUNet model.
    
    Args:
        compression_ratio: Î± value (e.g., 0.5 for 50% compression)
        upsample_ratio: Î² value (e.g., 4.0 for 4Ã— upsampling)
        hidden_dim: Feature dimension
        num_transformer_blocks: Number of transformer blocks
        with_dimension_handler: Whether to wrap with dimension handler
        
    Returns:
        Unified-CUNet model instance
    """
    model = UnifiedCUNet(
        point_dim=3,
        hidden_dim=hidden_dim,
        cs_dim=32,
        feature_dim=64,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=8,
        dropout=0.1,
        compression_ratio=compression_ratio,
        upsample_ratio=upsample_ratio
    )
    
    if with_dimension_handler:
        model = DimensionHandler(model, name='Unified-CUNet')
    
    return model


class UnifiedCUNetMLP(UnifiedCUNet):
    """
    MLP-variant of Unified-CUNet for ablation study.

    Identical to UnifiedCUNet in every respect except that the
    transformer_blocks ModuleList is replaced with MLPBlocks.
    This isolates the contribution of global self-attention:
    if performance drops significantly, it confirms that cross-point
    attention — not just representational capacity — is responsible
    for the surface-preservation advantage.

    Used for: Reviewer 2 Comment 1 (transformer vs. MLP ablation).
    """
    def __init__(
        self,
        point_dim=3,
        hidden_dim=64,
        cs_dim=32,
        feature_dim=64,
        num_transformer_blocks=4,
        dropout=0.1,
        compression_ratio=0.5,
        upsample_ratio=4.0
    ):
        # Call UnifiedCUNet __init__ — this creates self.transformer_blocks
        # with real TransformerBlocks.  We immediately replace them below.
        super(UnifiedCUNetMLP, self).__init__(
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            cs_dim=cs_dim,
            feature_dim=feature_dim,
            num_transformer_blocks=num_transformer_blocks,
            num_heads=8,          # unused after replacement
            dropout=dropout,
            compression_ratio=compression_ratio,
            upsample_ratio=upsample_ratio
        )

        # Replace TransformerBlocks with capacity-matched MLPBlocks
        self.transformer_blocks = nn.ModuleList([
            MLPBlock(hidden_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

    def get_config(self):
        cfg = super().get_config()
        cfg['name'] = 'Unified-CUNet-MLP'
        cfg['block_type'] = 'MLP (ablation)'
        return cfg


def create_unified_cunet_mlp(
    compression_ratio=0.5,
    upsample_ratio=4.0,
    hidden_dim=64,
    num_transformer_blocks=4,
    with_dimension_handler=True
):
    """
    Factory function to create the MLP-variant of Unified-CUNet.

    Drop-in replacement for create_unified_cunet(); identical call signature
    so it can be swapped into train.py / evaluate.py without other changes.

    Usage in train.py:
        from models.unified_cunet import create_unified_cunet_mlp
        model = create_unified_cunet_mlp(
            compression_ratio=args.compression_ratio,
            upsample_ratio=args.upsample_ratio
        )
    """
    model = UnifiedCUNetMLP(
        point_dim=3,
        hidden_dim=hidden_dim,
        cs_dim=32,
        feature_dim=64,
        num_transformer_blocks=num_transformer_blocks,
        dropout=0.1,
        compression_ratio=compression_ratio,
        upsample_ratio=upsample_ratio
    )

    if with_dimension_handler:
        model = DimensionHandler(model, name='Unified-CUNet-MLP')

    return model


if __name__ == "__main__":
    print("Testing Unified-CUNet...")

    model = create_unified_cunet(compression_ratio=0.75, upsample_ratio=4.0)

    batch_size = 4
    num_points = 2048
    x = torch.randn(batch_size, 3, num_points)
    print(f"Input shape: {x.shape}")

    output = model(x, mode='full')
    print(f"Output shape (full mode): {output.shape}")
    print(f"Expected output points: {int(num_points * 0.75 * 4.0)}")

    compressed = model(x, mode='compress')
    print(f"Compressed shape: {compressed.shape}")
    print("\nModel configuration:")
    print(model.get_config())

    # --------------------------------------------------------
    # MLP-variant ablation test
    # --------------------------------------------------------
    print("\n" + "="*50)
    print("Testing Unified-CUNet-MLP (ablation variant)...")
    mlp_model = create_unified_cunet_mlp(compression_ratio=0.75, upsample_ratio=4.0)
    mlp_output = mlp_model(x, mode='full')
    print(f"MLP variant output shape: {mlp_output.shape}")
    print(f"MLP variant config: {mlp_model.get_config()}")

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\nParameter counts:")
    print(f"  Unified-CUNet (transformer): {count_params(model):,}")
    print(f"  Unified-CUNet-MLP:           {count_params(mlp_model):,}")

    print("\n\u2713 All tests passed!")
