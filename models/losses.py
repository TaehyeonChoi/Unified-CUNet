"""
Loss Functions for Unified-CUNet Training

Implements the total loss function from Equation (15) in the paper:
    L_T = λ_CD · L_CD + λ_S · L_S + λ_D · L_D + λ_E · L_E + λ_n · L_n

Where:
    - L_CD: Chamfer Distance loss
    - L_S: Smoothness loss  
    - L_D: Distribution loss
    - L_E: Edge loss
    - L_n: Normal consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def chamfer_distance(pred, target, return_components=False):
    """
    Compute Chamfer Distance between two point clouds.
    
    Chamfer Distance measures bidirectional point-to-point distances,
    providing a symmetric measure of geometric similarity.
    
    Args:
        pred: Predicted point cloud [B, 3, N] or [B, N, 3]
        target: Target point cloud [B, 3, M] or [B, M, 3]
        return_components: Whether to return individual components
        
    Returns:
        Chamfer distance (scalar or tuple if return_components=True)
    """
    # Ensure [B, N, 3] format
    if pred.size(1) == 3 and pred.size(2) != 3:
        pred = pred.transpose(1, 2).contiguous()
    if target.size(1) == 3 and target.size(2) != 3:
        target = target.transpose(1, 2).contiguous()
    
    # Compute pairwise distances
    dist = torch.cdist(pred, target, p=2)  # [B, N, M]
    
    # Forward: pred → target (closest target for each predicted point)
    min_dist_pred_to_target = torch.min(dist, dim=2)[0]  # [B, N]
    loss_pred_to_target = torch.mean(min_dist_pred_to_target)
    
    # Backward: target → pred (closest predicted for each target point)
    min_dist_target_to_pred = torch.min(dist, dim=1)[0]  # [B, M]
    loss_target_to_pred = torch.mean(min_dist_target_to_pred)
    
    chamfer_dist = loss_pred_to_target + loss_target_to_pred
    
    if return_components:
        return chamfer_dist, loss_pred_to_target, loss_target_to_pred
    return chamfer_dist


def smoothness_loss(points, k=16):
    """
    Compute smoothness loss to encourage locally smooth surfaces.
    
    Measures variance of distances to k-nearest neighbors,
    encouraging uniform local point distributions.
    
    Args:
        points: Point cloud [B, 3, N] or [B, N, 3]
        k: Number of nearest neighbors
        
    Returns:
        Smoothness loss value
    """
    # Ensure [B, N, 3] format
    if points.size(1) == 3 and points.size(2) != 3:
        points = points.transpose(1, 2).contiguous()
    
    B, N, _ = points.shape
    k = min(k, N - 1)
    
    # Compute pairwise distances
    dist = torch.cdist(points, points)  # [B, N, N]
    
    # Get k-nearest neighbors (excluding self)
    knn_dist, _ = torch.topk(dist, k + 1, dim=2, largest=False)
    knn_dist = knn_dist[:, :, 1:]  # Exclude self [B, N, k]
    
    # Compute variance of neighbor distances per point
    mean_dist = knn_dist.mean(dim=2, keepdim=True)
    variance = ((knn_dist - mean_dist) ** 2).mean(dim=2)
    
    return variance.mean()


def distribution_loss(points, k=16):
    """
    Compute distribution loss for uniform point spacing.
    
    Encourages points to be evenly distributed across the surface
    by penalizing variance in local point densities.
    
    Args:
        points: Point cloud [B, 3, N] or [B, N, 3]
        k: Number of nearest neighbors
        
    Returns:
        Distribution loss value
    """
    # Ensure [B, N, 3] format
    if points.size(1) == 3 and points.size(2) != 3:
        points = points.transpose(1, 2).contiguous()
    
    B, N, _ = points.shape
    k = min(k, N - 1)
    
    # Compute pairwise distances
    dist = torch.cdist(points, points)  # [B, N, N]
    
    # Get k-nearest neighbors
    knn_dist, _ = torch.topk(dist, k + 1, dim=2, largest=False)
    knn_dist = knn_dist[:, :, 1:]  # Exclude self
    
    # Mean distance to k-nearest neighbors for each point
    local_density = knn_dist.mean(dim=2)  # [B, N]
    
    # Variance of local densities (lower = more uniform)
    density_variance = torch.var(local_density, dim=1).mean()
    
    return density_variance


def edge_loss(pred, target, threshold_percentile=90):
    """
    Compute edge/feature preservation loss.
    
    Focuses on high-curvature regions (edges, corners) to ensure
    these geometric features are preserved during processing.
    
    Args:
        pred: Predicted point cloud [B, 3, N] or [B, N, 3]
        target: Target point cloud [B, 3, M] or [B, M, 3]
        threshold_percentile: Percentile for edge point selection
        
    Returns:
        Edge preservation loss value
    """
    # Ensure [B, N, 3] format
    if pred.size(1) == 3 and pred.size(2) != 3:
        pred = pred.transpose(1, 2).contiguous()
    if target.size(1) == 3 and target.size(2) != 3:
        target = target.transpose(1, 2).contiguous()
    
    B = pred.size(0)
    total_loss = 0.0
    
    for b in range(B):
        # Estimate curvature for target points
        target_curvature = estimate_curvature(target[b])
        
        # Select high-curvature (edge) points
        threshold = torch.quantile(target_curvature, threshold_percentile / 100.0)
        edge_mask = target_curvature >= threshold
        edge_points = target[b][edge_mask]
        
        if edge_points.size(0) > 0:
            # Compute distance from predicted points to edge points
            dist = torch.cdist(pred[b:b+1], edge_points.unsqueeze(0))
            min_dist = dist.min(dim=1)[0]
            total_loss += min_dist.mean()
    
    return total_loss / B


def estimate_curvature(points, k=20):
    """
    Estimate local curvature at each point using PCA on neighborhood.
    
    Args:
        points: Point cloud [N, 3]
        k: Neighborhood size
        
    Returns:
        Curvature values [N]
    """
    N = points.size(0)
    k = min(k, N - 1)
    
    # Compute pairwise distances
    dist = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
    
    # Get k-nearest neighbors
    _, knn_idx = torch.topk(dist, k + 1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]  # Exclude self
    
    curvatures = torch.zeros(N, device=points.device)
    
    for i in range(N):
        neighbors = points[knn_idx[i]]  # [k, 3]
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        
        # PCA via SVD
        try:
            _, S, _ = torch.svd(centered)
            # Curvature approximated by smallest eigenvalue ratio
            curvatures[i] = S[2] / (S.sum() + 1e-8)
        except:
            curvatures[i] = 0.0
    
    return curvatures


def normal_consistency_loss(pred, target, k=20):
    """
    Compute normal consistency loss.
    
    Measures how well surface normals are preserved between
    predicted and target point clouds.
    
    Args:
        pred: Predicted point cloud [B, 3, N] or [B, N, 3]
        target: Target point cloud [B, 3, M] or [B, M, 3]
        k: Neighborhood size for normal estimation
        
    Returns:
        Normal consistency loss (1 - consistency, so lower is better)
    """
    # Ensure [B, N, 3] format
    if pred.size(1) == 3 and pred.size(2) != 3:
        pred = pred.transpose(1, 2).contiguous()
    if target.size(1) == 3 and target.size(2) != 3:
        target = target.transpose(1, 2).contiguous()
    
    B = pred.size(0)
    total_consistency = 0.0
    
    for b in range(B):
        # Estimate normals
        pred_normals = estimate_normals(pred[b], k)
        target_normals = estimate_normals(target[b], k)
        
        # Find correspondences (closest points)
        dist = torch.cdist(pred[b:b+1], target[b:b+1]).squeeze(0)
        _, closest_idx = dist.min(dim=1)
        
        # Compute normal consistency (absolute dot product)
        corresponding_normals = target_normals[closest_idx]
        consistency = torch.abs((pred_normals * corresponding_normals).sum(dim=1))
        total_consistency += consistency.mean()
    
    # Return as loss (1 - consistency)
    return 1.0 - (total_consistency / B)


def estimate_normals(points, k=20):
    """
    Estimate surface normals using PCA on local neighborhoods.
    
    Args:
        points: Point cloud [N, 3]
        k: Neighborhood size
        
    Returns:
        Normal vectors [N, 3]
    """
    N = points.size(0)
    k = min(k, N - 1)
    
    # Compute pairwise distances
    dist = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
    
    # Get k-nearest neighbors
    _, knn_idx = torch.topk(dist, k + 1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]  # Exclude self
    
    normals = torch.zeros(N, 3, device=points.device)
    
    for i in range(N):
        neighbors = points[knn_idx[i]]  # [k, 3]
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        
        # PCA via SVD - normal is eigenvector with smallest eigenvalue
        try:
            _, _, Vh = torch.linalg.svd(centered)
            normals[i] = Vh[2]  # Third row = smallest eigenvalue direction
        except:
            normals[i] = torch.tensor([0., 0., 1.], device=points.device)
    
    # Normalize
    normals = F.normalize(normals, dim=1)
    
    return normals


def repulsion_loss(points, k=16, h=0.03):
    """
    Compute repulsion loss to prevent point clustering.
    
    Encourages points to maintain minimum separation,
    avoiding degenerate point distributions.
    
    Args:
        points: Point cloud [B, 3, N] or [B, N, 3]
        k: Number of nearest neighbors
        h: Repulsion bandwidth parameter
        
    Returns:
        Repulsion loss value
    """
    # Ensure [B, N, 3] format
    if points.size(1) == 3 and points.size(2) != 3:
        points = points.transpose(1, 2).contiguous()
    
    B, N, _ = points.shape
    k = min(k, N - 1)
    
    # Compute pairwise distances
    dist = torch.cdist(points, points)  # [B, N, N]
    
    # Get k-nearest neighbors (excluding self)
    knn_dist, _ = torch.topk(dist, k + 1, dim=2, largest=False)
    knn_dist = knn_dist[:, :, 1:]  # Exclude self
    
    # Repulsion term: exp(-d^2 / h^2)
    repulsion = torch.exp(-knn_dist.pow(2) / (h ** 2))
    
    return repulsion.mean()


class UnifiedCUNetLoss(nn.Module):
    """
    Complete loss function for Unified-CUNet training.
    
    Implements Equation (15) from the paper:
        L_T = λ_CD · L_CD + λ_S · L_S + λ_D · L_D + λ_E · L_E + λ_n · L_n
    
    Args:
        lambda_cd: Weight for Chamfer Distance loss
        lambda_smoothness: Weight for smoothness loss
        lambda_distribution: Weight for distribution loss
        lambda_edge: Weight for edge preservation loss
        lambda_normal: Weight for normal consistency loss
    """
    def __init__(
        self,
        lambda_cd=1.0,
        lambda_smoothness=0.1,
        lambda_distribution=0.5,
        lambda_edge=0.3,
        lambda_normal=0.2
    ):
        super(UnifiedCUNetLoss, self).__init__()
        
        self.lambda_cd = lambda_cd
        self.lambda_smoothness = lambda_smoothness
        self.lambda_distribution = lambda_distribution
        self.lambda_edge = lambda_edge
        self.lambda_normal = lambda_normal
        
        # For tracking individual loss components
        self.loss_components = {}
    
    def forward(self, pred, target):
        """
        Compute total loss.
        
        Args:
            pred: Predicted point cloud [B, 3, N]
            target: Target point cloud [B, 3, M]
            
        Returns:
            Total weighted loss
        """
        # L_CD: Chamfer Distance
        loss_cd = chamfer_distance(pred, target)
        
        # L_S: Smoothness
        loss_smoothness = smoothness_loss(pred)
        
        # L_D: Distribution
        loss_distribution = distribution_loss(pred)
        
        # L_E: Edge preservation
        loss_edge = edge_loss(pred, target)
        
        # L_n: Normal consistency
        loss_normal = normal_consistency_loss(pred, target)
        
        # Total loss (Equation 15)
        total_loss = (
            self.lambda_cd * loss_cd +
            self.lambda_smoothness * loss_smoothness +
            self.lambda_distribution * loss_distribution +
            self.lambda_edge * loss_edge +
            self.lambda_normal * loss_normal
        )
        
        # Store components for logging
        self.loss_components = {
            'total': total_loss.item(),
            'chamfer': loss_cd.item(),
            'smoothness': loss_smoothness.item(),
            'distribution': loss_distribution.item(),
            'edge': loss_edge.item(),
            'normal': loss_normal.item()
        }
        
        return total_loss
    
    def get_loss_components(self):
        """Return dictionary of individual loss components."""
        return self.loss_components


class SimpleLoss(nn.Module):
    """
    Simplified loss function using only Chamfer Distance.
    
    Useful for initial training or comparison experiments.
    """
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.loss_components = {}
    
    def forward(self, pred, target):
        loss = chamfer_distance(pred, target)
        self.loss_components = {'chamfer': loss.item(), 'total': loss.item()}
        return loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 2
    num_points_pred = 512
    num_points_target = 2048
    
    pred = torch.randn(batch_size, 3, num_points_pred)
    target = torch.randn(batch_size, 3, num_points_target)
    
    # Test individual losses
    print(f"Chamfer Distance: {chamfer_distance(pred, target):.6f}")
    print(f"Smoothness Loss: {smoothness_loss(pred):.6f}")
    print(f"Distribution Loss: {distribution_loss(pred):.6f}")
    print(f"Repulsion Loss: {repulsion_loss(pred):.6f}")
    
    # Test combined loss
    loss_fn = UnifiedCUNetLoss()
    total_loss = loss_fn(pred, target)
    print(f"\nTotal Loss: {total_loss:.6f}")
    print("Loss Components:", loss_fn.get_loss_components())
    
    print("\n✓ All loss function tests passed!")
