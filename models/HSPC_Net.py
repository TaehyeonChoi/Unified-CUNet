"""
HSPC-Net: A Hierarchical Shape-Preserving Completion Network
for Point Cloud Upsampling

Adapted from:
    Jiang Y, Fan H, Zhu H (2025).
    "HSPC-Net: A hierarchical shape-preserving completion network
     for machine part point cloud completion."
    PLOS ONE. https://doi.org/10.1371/journal.pone.0330033

Architecture overview:
    1. Hierarchical Encoder  — multi-scale Set Abstraction layers capture
                               features at coarse / medium / fine resolutions.
    2. Multi-Receptive Field Transformer (MRFT)
                             — self-attention blocks whose neighbourhood
                               radii vary per head, giving the model both
                               local and global context simultaneously.
    3. Cross-Modal Geometric Fusion (CMGF)
                             — explicit geometric descriptors (surface normals
                               estimated from k-NN, local curvature) are fused
                               with the transformer features to keep the network
                               surface-aware throughout.
    4. Shape-Preserving Decoder
                             — hierarchical feature propagation from coarse to
                               fine scale, followed by coordinate reconstruction.

The original paper targets point cloud *completion* (restoring missing regions).
Here the model is repurposed as an *upsampling* network following the same
sequential pipeline used for PU-Net, PU-GACNet, and ADCoSNet baselines:

    Input (N pts)  -->  [FPS compress to N*alpha]  -->  HSPC-Net  -->  Output (N*alpha*beta pts)

Usage:
    model = HSPCNet(up_ratio=4)
    output = model(x)   # x: [B, N, 3]  ->  [B, N*up_ratio, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Utility: k-NN indexing
# ---------------------------------------------------------------------------

def knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return indices of k nearest neighbours for each point.

    Args:
        x:  [B, N, 3]
        k:  number of neighbours (self excluded)
    Returns:
        idx: [B, N, k]
    """
    sq_dist = torch.cdist(x, x, p=2)
    sq_dist = sq_dist + torch.eye(x.size(1), device=x.device).unsqueeze(0) * 1e9
    idx = sq_dist.topk(k, dim=-1, largest=False).indices   # [B, N, k]
    return idx


def gather_neighbours(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather neighbour features.

    Args:
        x:   [B, N, C]
        idx: [B, N, k]
    Returns:
        neighbours: [B, N, k, C]
    """
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_expanded = idx.unsqueeze(-1).expand(B, N, k, C)    # [B, N, k, C]
    x_expanded   = x.unsqueeze(2).expand(B, N, N, C)       # [B, N, N, C]
    return torch.gather(x_expanded, 2, idx_expanded)        # [B, N, k, C]


# ---------------------------------------------------------------------------
# Local geometric descriptor (surface normal + curvature proxy)
# ---------------------------------------------------------------------------

def estimate_local_geometry(x: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    Estimate per-point geometric descriptors from k-NN neighbourhood.

    Computes:
        - PCA-based surface normal approximation  (3 dims)
        - Mean distance to neighbours             (1 dim)
        - Variance of distances (curvature proxy) (1 dim)

    Args:
        x: [B, N, 3]
        k: neighbourhood size
    Returns:
        geo: [B, N, 5]
    """
    B, N, _ = x.shape
    k = min(k, N - 1)
    idx  = knn_idx(x, k)                           # [B, N, k]
    nbrs = gather_neighbours(x, idx)               # [B, N, k, 3]

    centroid = nbrs.mean(dim=2, keepdim=True)      # [B, N, 1, 3]
    centered = nbrs - centroid                     # [B, N, k, 3]

    # Covariance matrix [B*N, 3, 3]
    cov = torch.bmm(
        centered.reshape(B * N, k, 3).transpose(1, 2),
        centered.reshape(B * N, k, 3),
    ) / (k - 1)

    try:
        _, _, Vt = torch.linalg.svd(cov)
        normal = Vt[:, -1, :].reshape(B, N, 3)    # smallest eigenvector
    except Exception:
        normal = torch.zeros(B, N, 3, device=x.device)

    diff      = nbrs - x.unsqueeze(2)             # [B, N, k, 3]
    dists     = diff.norm(dim=-1)                  # [B, N, k]
    mean_dist = dists.mean(dim=-1, keepdim=True)   # [B, N, 1]
    var_dist  = dists.var(dim=-1, keepdim=True)    # [B, N, 1]

    geo = torch.cat([normal, mean_dist, var_dist], dim=-1)  # [B, N, 5]
    return geo.detach()


# ---------------------------------------------------------------------------
# Set Abstraction layer (local feature pooling — no FPS; full point set kept)
# ---------------------------------------------------------------------------

class SetAbstractionLayer(nn.Module):
    """
    EdgeConv-style Set Abstraction: aggregate k-NN edge features via
    a shared MLP then max-pool over the neighbourhood.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]   (first 3 channels used as coordinates for kNN)
        Returns:
            out: [B, N, out_channels]
        """
        B, N, C = x.shape
        k = min(self.k, N - 1)
        idx  = knn_idx(x[..., :3], k)             # [B, N, k]
        nbrs = gather_neighbours(x, idx)           # [B, N, k, C]

        x_rep = x.unsqueeze(2).expand(B, N, k, C)
        edge  = torch.cat([x_rep, nbrs - x_rep], dim=-1)  # [B, N, k, 2C]

        edge = edge.permute(0, 3, 2, 1)            # [B, 2C, k, N]
        feat = self.mlp(edge)                      # [B, out, k, N]
        feat = feat.max(dim=2).values              # [B, out, N]
        return feat.transpose(1, 2)                # [B, N, out]


# ---------------------------------------------------------------------------
# Multi-Receptive Field Transformer (MRFT)
# ---------------------------------------------------------------------------

class MultiReceptiveFieldAttention(nn.Module):
    """
    Multi-head self-attention where each head uses a different local
    neighbourhood size (k_list), spanning fine to coarse receptive fields.
    """

    def __init__(self, d_model: int, num_heads: int = 4,
                 k_list: tuple = (8, 16, 32, 64)):
        super().__init__()
        assert len(k_list) == num_heads
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.k_list    = k_list
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      [B, N, d_model]
            coords: [B, N, 3]
        Returns:
            out: [B, N, d_model]
        """
        B, N, D = x.shape
        Hd = self.head_dim

        Q = self.q_proj(x).reshape(B, N, self.num_heads, Hd).transpose(1, 2)
        K = self.k_proj(x).reshape(B, N, self.num_heads, Hd).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, self.num_heads, Hd).transpose(1, 2)

        head_outputs = []
        for h in range(self.num_heads):
            k = min(self.k_list[h], N - 1)
            idx  = knn_idx(coords, k)                          # [B, N, k]
            idx_e = idx.unsqueeze(-1).expand(B, N, k, Hd)

            K_h = K[:, h]                                      # [B, N, Hd]
            V_h = V[:, h]
            K_nbr = torch.gather(K_h.unsqueeze(2).expand(B, N, N, Hd), 2, idx_e)
            V_nbr = torch.gather(V_h.unsqueeze(2).expand(B, N, N, Hd), 2, idx_e)

            attn  = (Q[:, h].unsqueeze(2) * K_nbr).sum(-1) * self.scale  # [B, N, k]
            attn  = F.softmax(attn, dim=-1)
            out_h = (attn.unsqueeze(-1) * V_nbr).sum(dim=2)               # [B, N, Hd]
            head_outputs.append(out_h)

        out = torch.cat(head_outputs, dim=-1)                  # [B, N, D]
        return self.out_proj(out)


class MRFTBlock(nn.Module):
    """Single MRFT block: MRFA → Add & LayerNorm → FFN → Add & LayerNorm."""

    def __init__(self, d_model: int, num_heads: int = 4,
                 k_list: tuple = (8, 16, 32, 64), ffn_ratio: int = 2):
        super().__init__()
        self.attn  = MultiReceptiveFieldAttention(d_model, num_heads, k_list)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_model * ffn_ratio, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, coords))
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Cross-Modal Geometric Fusion (CMGF)
# ---------------------------------------------------------------------------

class CrossModalGeometricFusion(nn.Module):
    """
    Fuse transformer (semantic) features with local geometric descriptors
    using a learnable gating mechanism.

    gate = sigmoid( Linear([feat, geo_proj]) )
    out  = Linear( feat + gate * geo_proj )
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.geo_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.gate     = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, feat: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, N, d_model]
            geo:  [B, N, 5]
        Returns:
            fused: [B, N, d_model]
        """
        g    = self.geo_proj(geo)
        gate = self.gate(torch.cat([feat, g], dim=-1))
        return self.out_proj(feat + gate * g)


# ---------------------------------------------------------------------------
# Hierarchical Encoder
# ---------------------------------------------------------------------------

class HierarchicalEncoder(nn.Module):
    """
    Three-level encoder:
        Level 1 (fine)   : SA(3       → 64,      k=16) → proj → d_model
        Level 2 (medium) : SA(d_model → d_model, k=24) + MRFT + CMGF
        Level 3 (coarse) : SA(d_model → d_model, k=32) + MRFT + CMGF

    sa2 and sa3 receive already-projected d_model-wide features (f1, f1+f2),
    so their in_channels must equal d_model, not the raw SA output sizes.
    All levels share the same projected dimension d_model so that
    the decoder can fuse them directly.
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        # sa1: input is raw 3-D coordinates
        self.sa1 = SetAbstractionLayer(3,       64,      k=16)
        # sa2/sa3: input is already-projected d_model-wide features
        self.sa2 = SetAbstractionLayer(d_model, d_model, k=24)
        self.sa3 = SetAbstractionLayer(d_model, d_model, k=32)

        self.proj1 = nn.Linear(64,      d_model)
        self.proj2 = nn.Linear(d_model, d_model)
        self.proj3 = nn.Linear(d_model, d_model)

        self.mrft1 = MRFTBlock(d_model, num_heads, k_list=(4,  8,  16, 32))
        self.mrft2 = MRFTBlock(d_model, num_heads, k_list=(8,  16, 32, 64))
        self.mrft3 = MRFTBlock(d_model, num_heads, k_list=(8,  16, 48, 96))

        self.cmgf1 = CrossModalGeometricFusion(d_model)
        self.cmgf2 = CrossModalGeometricFusion(d_model)
        self.cmgf3 = CrossModalGeometricFusion(d_model)

    def forward(self, coords: torch.Tensor):
        """
        Args:
            coords: [B, N, 3]
        Returns:
            f1, f2, f3: [B, N, d_model]  — encoder features at three levels
            geo:        [B, N, 5]        — geometric descriptors
        """
        geo = estimate_local_geometry(coords, k=16)

        f1 = self.proj1(self.sa1(coords))
        f1 = self.cmgf1(self.mrft1(f1, coords), geo)

        f2 = self.proj2(self.sa2(f1))
        f2 = self.cmgf2(self.mrft2(f2, coords), geo)

        f3 = self.proj3(self.sa3(f1 + f2))          # skip connection
        f3 = self.cmgf3(self.mrft3(f3, coords), geo)

        return f1, f2, f3, geo


# ---------------------------------------------------------------------------
# Shape-Preserving Decoder
# ---------------------------------------------------------------------------

class ShapePreservingDecoder(nn.Module):
    """
    Hierarchical feature propagation decoder.

    Fuses features from all three encoder levels, refines them with one
    more MRFT block, then generates up_ratio new point offsets per input
    point via branched MLPs (multi-branch expansion from PU-Net).
    """

    def __init__(self, d_model: int = 128, up_ratio: int = 4,
                 num_heads: int = 4, k_list: tuple = (8, 16, 32, 64)):
        super().__init__()
        self.up_ratio = up_ratio

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.refine = MRFTBlock(d_model, num_heads, k_list)

        # One branch per upsampled point  (outputs offset from input point)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
            ) for _ in range(up_ratio)
        ])

    def forward(self, f1: torch.Tensor, f2: torch.Tensor,
                f3: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1, f2, f3: [B, N, d_model]
            coords:     [B, N, 3]
        Returns:
            upsampled: [B, N*up_ratio, 3]
        """
        fused  = self.fuse(torch.cat([f1, f2, f3], dim=-1))   # [B, N, d_model]
        fused  = self.refine(fused, coords)

        points = [coords + branch(fused) for branch in self.branches]  # R * [B, N, 3]
        out    = torch.stack(points, dim=2).reshape(
                     coords.size(0), coords.size(1) * self.up_ratio, 3)
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HSPCNet(nn.Module):
    """
    HSPC-Net adapted for point cloud upsampling.

    Args:
        up_ratio:  upsampling factor β  (default 4)
        d_model:   internal feature dimension  (default 128)
        num_heads: number of attention heads per MRFT block  (default 4)
    """

    def __init__(self, up_ratio: int = 4, d_model: int = 128,
                 num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.up_ratio = up_ratio
        self.encoder  = HierarchicalEncoder(d_model=d_model, num_heads=num_heads)
        self.decoder  = ShapePreservingDecoder(
            d_model=d_model, up_ratio=up_ratio,
            num_heads=num_heads, k_list=(8, 16, 32, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3]  (FPS-compressed input)
        Returns:
            upsampled: [B, N*up_ratio, 3]
        """
        if x.dim() == 3 and x.size(2) != 3:          # [B, 3, N] -> [B, N, 3]
            x = x.transpose(1, 2).contiguous()
        coords = x[..., :3].contiguous()
        f1, f2, f3, _ = self.encoder(coords)
        return self.decoder(f1, f2, f3, coords)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class HSPCLoss(nn.Module):
    """
    Shape-preserving loss:
        L = lambda_cd * L_cd  +  lambda_nc * L_nc  +  lambda_rep * L_rep

    L_cd  — bidirectional Chamfer Distance
    L_nc  — normal consistency (angular deviation between predicted and
             target surface normals estimated from k-NN)
    L_rep — repulsion loss (prevents point clustering)
    """

    def __init__(self, lambda_cd: float = 1.0,
                 lambda_nc: float  = 0.2,
                 lambda_rep: float = 0.1):
        super().__init__()
        self.lambda_cd  = lambda_cd
        self.lambda_nc  = lambda_nc
        self.lambda_rep = lambda_rep

    @staticmethod
    def chamfer_distance(pred: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(pred, target)
        return dist.min(2).values.mean() + dist.min(1).values.mean()

    @staticmethod
    def normal_consistency(pred: torch.Tensor,
                           target: torch.Tensor,
                           k: int = 8) -> torch.Tensor:
        def normals(pts):
            B, N, _ = pts.shape
            k_ = min(k, N - 1)
            idx  = knn_idx(pts, k_)
            nbrs = gather_neighbours(pts, idx)
            centered = nbrs - pts.unsqueeze(2)
            cov = torch.bmm(
                centered.reshape(B * N, k_, 3).transpose(1, 2),
                centered.reshape(B * N, k_, 3),
            ) / (k_ - 1)
            try:
                _, _, Vt = torch.linalg.svd(cov)
                n = Vt[:, -1, :].reshape(B, N, 3)
            except Exception:
                n = torch.zeros(B, N, 3, device=pts.device)
            return F.normalize(n, dim=-1)

        n_p = normals(pred)
        n_t = normals(target)
        nn_idx = torch.cdist(pred, target).argmin(-1)          # [B, N_p]
        B, Np, _ = pred.shape
        n_t_matched = torch.gather(n_t, 1,
                          nn_idx.unsqueeze(-1).expand(B, Np, 3))
        return (1.0 - (n_p * n_t_matched).sum(-1).abs()).mean()

    @staticmethod
    def repulsion_loss(pred: torch.Tensor,
                       k: int = 8, h: float = 0.03) -> torch.Tensor:
        dist = torch.cdist(pred, pred)
        dist = dist + torch.eye(pred.size(1), device=pred.device).unsqueeze(0) * 1e9
        knn_d = dist.topk(k, dim=-1, largest=False).values
        return torch.exp(-(knn_d / h) ** 2).mean()

    @staticmethod
    def _to_bnc(t: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is [B, N, 3]. Accepts BCN [B, 3, N] and BNC [B, N, 3]."""
        if t.dim() == 3 and t.size(1) == 3 and t.size(2) != 3:
            return t.transpose(1, 2).contiguous()   # [B, 3, N] -> [B, N, 3]
        return t.contiguous()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # Normalise both to [B, N, 3] regardless of what the caller passes
        pred   = self._to_bnc(pred)
        target = self._to_bnc(target)
        return (self.lambda_cd  * self.chamfer_distance(pred, target) +
                self.lambda_nc  * self.normal_consistency(pred, target) +
                self.lambda_rep * self.repulsion_loss(pred))


# ---------------------------------------------------------------------------
# Quick sanity check  (python HSPC_Net.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, N, up = 2, 512, 4
    x = torch.randn(B, N, 3).to(device)

    model = HSPCNet(up_ratio=up, d_model=128, num_heads=4).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (B, N * up, 3), f"Shape error: {out.shape}"
    print(f"Input  : {list(x.shape)}")
    print(f"Output : {list(out.shape)}  ✓")

    loss_fn = HSPCLoss()
    target  = torch.randn(B, N * up, 3).to(device)
    with torch.no_grad():
        loss = loss_fn(out, target)
    print(f"Loss   : {loss.item():.6f}  ✓")
    print("\nSanity check passed.")
