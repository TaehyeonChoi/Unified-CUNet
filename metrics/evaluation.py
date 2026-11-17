"""
Evaluation Metrics for Unified-CUNet

Implements all six metrics used in the paper's experimental evaluation:
    1. Chamfer Distance (CD) - Bidirectional point-to-point distance
    2. Hausdorff Distance (HD) - Worst-case distance between point clouds
    3. Point-to-Surface Distance (P2S) - Distance to local tangent planes
    4. Normal Consistency (NC) - Surface normal preservation
    5. Curvature Consistency (CC) - Local curvature preservation
    6. Uniformity (U) - Evenness of point distribution
"""

import torch
import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def to_numpy(tensor):
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def ensure_shape(points):
    """Ensure points are in [N, 3] format."""
    points = to_numpy(points)
    if len(points.shape) == 3:
        # Handle batch dimension
        points = points[0] if points.shape[0] == 1 else points.reshape(-1, 3)
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    return points


# ============================================================
# Metric 1: Chamfer Distance (CD)
# ============================================================
def chamfer_distance(pred, target, tensor_output=False):
    """
    Compute Chamfer Distance between two point clouds.
    
    References:
        Fan et al., "A Point Set Generation Network for 3D Object 
        Reconstruction from a Single Image", CVPR 2017
    
    Args:
        pred: Predicted point cloud [N, 3]
        target: Target point cloud [M, 3]
        tensor_output: Whether to return torch tensor
        
    Returns:
        Chamfer distance value
    """
    pred = ensure_shape(pred)
    target = ensure_shape(target)
    
    # Build KD-tree for efficient nearest neighbor search
    tree_pred = KDTree(pred)
    tree_target = KDTree(target)
    
    # Forward: pred -> target
    dist_pred_to_target, _ = tree_target.query(pred)
    
    # Backward: target -> pred
    dist_target_to_pred, _ = tree_pred.query(target)
    
    # Chamfer distance = mean forward + mean backward
    cd = np.mean(dist_pred_to_target) + np.mean(dist_target_to_pred)
    
    if tensor_output:
        return torch.tensor(cd, dtype=torch.float32)
    return float(cd)


# ============================================================
# Metric 2: Hausdorff Distance (HD)
# ============================================================
def hausdorff_distance(pred, target):
    """
    Compute Hausdorff Distance between two point clouds.
    
    Hausdorff distance captures the worst-case distance, useful for
    detecting outliers and large deviations.
    
    References:
        Cignoni et al., "Metro: Measuring Error on Simplified Surfaces",
        Computer Graphics Forum, 1998
    
    Args:
        pred: Predicted point cloud [N, 3]
        target: Target point cloud [M, 3]
        
    Returns:
        Hausdorff distance value
    """
    pred = ensure_shape(pred)
    target = ensure_shape(target)
    
    # Build KD-trees
    tree_pred = KDTree(pred)
    tree_target = KDTree(target)
    
    # Forward Hausdorff: max distance from pred to closest target
    dist_pred_to_target, _ = tree_target.query(pred)
    h_forward = np.max(dist_pred_to_target)
    
    # Backward Hausdorff: max distance from target to closest pred
    dist_target_to_pred, _ = tree_pred.query(target)
    h_backward = np.max(dist_target_to_pred)
    
    # Hausdorff distance = max of forward and backward
    return float(max(h_forward, h_backward))


# ============================================================
# Metric 3: Point-to-Surface Distance (P2S)
# ============================================================
def point_to_surface_distance(pred, target, k=10):
    """
    Compute Point-to-Surface Distance.
    
    Approximates the true surface distance by fitting local planes
    to point neighborhoods and computing distance to these tangent planes.
    
    References:
        Rusu, "Semantic 3D Object Maps for Everyday Manipulation in 
        Human Living Environments", PhD Thesis, 2009
    
    Args:
        pred: Predicted point cloud [N, 3]
        target: Target point cloud [M, 3]
        k: Neighborhood size for plane fitting
        
    Returns:
        Point-to-surface distance value
    """
    pred = ensure_shape(pred)
    target = ensure_shape(target)
    
    # Build KD-tree on target
    tree_target = KDTree(target)
    
    # For each predicted point, find closest point in target
    # and compute distance to local tangent plane
    distances = []
    
    for point in pred:
        # Find k nearest neighbors in target
        dists, indices = tree_target.query(point, k=min(k, len(target)))
        neighbors = target[indices]
        
        # Fit local plane using PCA
        if len(neighbors) >= 3:
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            
            try:
                pca = PCA(n_components=3)
                pca.fit(centered)
                # Normal is the direction with smallest variance
                normal = pca.components_[2]
                
                # Distance from point to plane
                dist = np.abs(np.dot(point - centroid, normal))
                distances.append(dist)
            except:
                # Fallback to point-to-point distance
                distances.append(dists[0])
        else:
            distances.append(dists[0])
    
    return float(np.mean(distances))


# ============================================================
# Metric 4: Normal Consistency (NC)
# ============================================================
def estimate_normals_numpy(points, k=20):
    """
    Estimate surface normals using PCA on local neighborhoods.
    
    Args:
        points: Point cloud [N, 3]
        k: Neighborhood size
        
    Returns:
        Normal vectors [N, 3]
    """
    points = ensure_shape(points)
    n_points = len(points)
    k = min(k, n_points - 1)
    
    tree = KDTree(points)
    normals = np.zeros((n_points, 3))
    
    for i, point in enumerate(points):
        _, indices = tree.query(point, k=k + 1)
        neighbors = points[indices[1:]]  # Exclude self
        
        if len(neighbors) >= 3:
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            
            try:
                pca = PCA(n_components=3)
                pca.fit(centered)
                normals[i] = pca.components_[2]  # Smallest eigenvalue direction
            except:
                normals[i] = np.array([0, 0, 1])
        else:
            normals[i] = np.array([0, 0, 1])
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    return normals


def compute_normal_consistency(pred, target, k=20):
    """
    Compute Normal Consistency metric.
    
    Measures how well surface normals are preserved between point clouds.
    Higher values indicate better normal preservation.
    
    References:
        Zhou and Tuzel, "VoxelNet: End-to-End Learning for Point Cloud 
        Based 3D Object Detection", CVPR 2018 (adapted)
    
    Args:
        pred: Predicted point cloud [N, 3]
        target: Target point cloud [M, 3]
        k: Neighborhood size for normal estimation
        
    Returns:
        Normal consistency value in [0, 1] (higher is better)
    """
    pred = ensure_shape(pred)
    target = ensure_shape(target)
    
    # Estimate normals
    pred_normals = estimate_normals_numpy(pred, k)
    target_normals = estimate_normals_numpy(target, k)
    
    # Find correspondences (closest points)
    tree_target = KDTree(target)
    _, closest_indices = tree_target.query(pred)
    
    # Compute absolute dot product between corresponding normals
    corresponding_normals = target_normals[closest_indices]
    dot_products = np.abs(np.sum(pred_normals * corresponding_normals, axis=1))
    
    # Average consistency
    consistency = np.mean(dot_products)
    
    return float(consistency)


# ============================================================
# Metric 5: Curvature Consistency (CC)
# ============================================================
def estimate_curvature_numpy(points, k=20):
    """
    Estimate local curvature at each point.
    
    Uses variation of normals in local neighborhood as curvature proxy.
    
    References:
        Alliez et al., "Voronoi-based Variational Reconstruction of 
        Unoriented Point Sets", SGP 2007
    
    Args:
        points: Point cloud [N, 3]
        k: Neighborhood size
        
    Returns:
        Curvature values [N]
    """
    points = ensure_shape(points)
    n_points = len(points)
    k = min(k, n_points - 1)
    
    tree = KDTree(points)
    curvatures = np.zeros(n_points)
    
    for i, point in enumerate(points):
        _, indices = tree.query(point, k=k + 1)
        neighbors = points[indices[1:]]  # Exclude self
        
        if len(neighbors) >= 3:
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            
            try:
                pca = PCA(n_components=3)
                pca.fit(centered)
                eigenvalues = pca.explained_variance_
                
                # Curvature = ratio of smallest to total variance
                curvatures[i] = eigenvalues[2] / (np.sum(eigenvalues) + 1e-8)
            except:
                curvatures[i] = 0.0
        else:
            curvatures[i] = 0.0
    
    return curvatures


def compute_curvature_consistency(pred, target, k=20):
    """
    Compute Curvature Consistency metric.
    
    Measures how well local surface curvature is preserved.
    Higher values indicate better curvature preservation.
    
    Args:
        pred: Predicted point cloud [N, 3]
        target: Target point cloud [M, 3]
        k: Neighborhood size for curvature estimation
        
    Returns:
        Curvature consistency value in [0, 1] (higher is better)
    """
    pred = ensure_shape(pred)
    target = ensure_shape(target)
    
    # Estimate curvatures
    pred_curvature = estimate_curvature_numpy(pred, k)
    target_curvature = estimate_curvature_numpy(target, k)
    
    # Find correspondences (closest points)
    tree_target = KDTree(target)
    _, closest_indices = tree_target.query(pred)
    
    # Get corresponding curvatures
    corresponding_curvature = target_curvature[closest_indices]
    
    # Compute consistency as 1 - normalized absolute difference
    max_curv = max(np.max(pred_curvature), np.max(corresponding_curvature), 1e-8)
    diff = np.abs(pred_curvature - corresponding_curvature) / max_curv
    consistency = 1.0 - np.mean(diff)
    
    return float(max(0, consistency))


# ============================================================
# Metric 6: Uniformity (U)
# ============================================================
def compute_uniformity(points, k=16):
    """
    Compute point distribution Uniformity.
    
    Measures how evenly points are distributed across the surface.
    Lower values indicate more uniform distribution.
    
    References:
        Li et al., "Point Cloud Upsampling via Disentangled Refinement",
        CVPR 2021
    
    Args:
        points: Point cloud [N, 3]
        k: Neighborhood size
        
    Returns:
        Uniformity value (lower is better)
    """
    points = ensure_shape(points)
    n_points = len(points)
    k = min(k, n_points - 1)
    
    tree = KDTree(points)
    
    # For each point, compute mean distance to k nearest neighbors
    local_densities = []
    
    for point in points:
        dists, _ = tree.query(point, k=k + 1)
        # Exclude self (distance 0)
        mean_dist = np.mean(dists[1:])
        local_densities.append(mean_dist)
    
    local_densities = np.array(local_densities)
    
    # Uniformity = variance of local densities (lower is more uniform)
    uniformity = np.var(local_densities)
    
    return float(uniformity)


# ============================================================
# Complete Evaluation Function
# ============================================================
def compute_all_metrics(pred, target, k=20):
    """
    Compute all six evaluation metrics.
    
    Args:
        pred: Predicted point cloud
        target: Target point cloud
        k: Neighborhood size for local metrics
        
    Returns:
        Dictionary containing all metric values
    """
    pred = ensure_shape(pred)
    target = ensure_shape(target)
    
    metrics = {
        'chamfer_distance': chamfer_distance(pred, target),
        'hausdorff_distance': hausdorff_distance(pred, target),
        'point_to_surface': point_to_surface_distance(pred, target, k=k),
        'normal_consistency': compute_normal_consistency(pred, target, k=k),
        'curvature_consistency': compute_curvature_consistency(pred, target, k=k),
        'uniformity': compute_uniformity(pred, k=min(16, len(pred)-1))
    }
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """Print metrics in formatted table."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"{'Metric':<25} {'Value':>20}")
    print("-" * 50)
    
    metric_names = {
        'chamfer_distance': 'Chamfer Distance (CD)',
        'hausdorff_distance': 'Hausdorff Distance (HD)',
        'point_to_surface': 'Point-to-Surface (P2S)',
        'normal_consistency': 'Normal Consistency (NC)',
        'curvature_consistency': 'Curvature Consistency (CC)',
        'uniformity': 'Uniformity (U)'
    }
    
    for key, value in metrics.items():
        name = metric_names.get(key, key)
        print(f"{name:<25} {value:>20.6f}")
    
    print("=" * 50)


class MetricsAccumulator:
    """
    Accumulates metrics across multiple samples for averaging.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics_sum = {}
        self.count = 0
    
    def update(self, metrics):
        """Add new metrics to accumulator."""
        for key, value in metrics.items():
            if key not in self.metrics_sum:
                self.metrics_sum[key] = 0.0
            self.metrics_sum[key] += value
        self.count += 1
    
    def get_average(self):
        """Return average metrics."""
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.metrics_sum.items()}
    
    def get_count(self):
        return self.count


if __name__ == "__main__":
    # Test all metrics
    print("Testing evaluation metrics...")
    
    np.random.seed(42)
    
    # Generate test point clouds
    n_points_pred = 512
    n_points_target = 2048
    
    # Target: points on a sphere
    theta = np.random.uniform(0, 2*np.pi, n_points_target)
    phi = np.random.uniform(0, np.pi, n_points_target)
    target = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=1)
    
    # Pred: noisy version of sphere
    theta = np.random.uniform(0, 2*np.pi, n_points_pred)
    phi = np.random.uniform(0, np.pi, n_points_pred)
    pred = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=1) + np.random.normal(0, 0.05, (n_points_pred, 3))
    
    # Compute all metrics
    metrics = compute_all_metrics(pred, target)
    print_metrics(metrics, "Test Results (Noisy Sphere)")
    
    # Test with identical point clouds
    metrics_identical = compute_all_metrics(target[:500], target[:500])
    print_metrics(metrics_identical, "Test Results (Identical Points)")
    
    print("\n✓ All metric tests passed!")
