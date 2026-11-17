import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelNet40Dataset(Dataset):
    """ModelNet40 Dataset loader"""
    def __init__(self, root_dir, split='train', num_points=2048, category=None, transform=None):
        """
        Args:
            root_dir: Path to ModelNet40 dataset root directory
            split: 'train' or 'test'
            num_points: Number of points to sample from each model
            category: If specified, only load objects from this category
            transform: Optional transform to apply to the point clouds
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform

        # Get all categories or use the specified one
        if category is not None:
            self.categories = [category]
        else:
            self.categories = [d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))]
            self.categories.sort()  # For deterministic behavior

        # Get all OFF files
        self.file_paths = []
        self.labels = []

        for i, category in enumerate(self.categories):
            category_split_dir = os.path.join(root_dir, category, split)
            if not os.path.exists(category_split_dir):
                print(f"Warning: Directory not found: {category_split_dir}")
                continue

            off_files = [f for f in os.listdir(category_split_dir) if f.endswith('.off')]
            self.file_paths.extend([os.path.join(category_split_dir, f) for f in off_files])
            self.labels.extend([i] * len(off_files))

        print(f"Loaded {len(self.file_paths)} {split} samples from {len(self.categories)} categories")

    def read_off(self, file_path):
        """Read OFF file and return vertices"""
        vertices = []
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # Skip the 'OFF' header
                if 'OFF' in first_line:
                    if len(first_line) > 3:
                        # Handle cases where vertex/face numbers are on the same line as OFF
                        counts = first_line[3:].strip().split()
                    else:
                        # Handle cases where vertex/face numbers are on the next line
                        counts = f.readline().strip().split()
                else:
                    counts = first_line.split()

                n_verts = int(counts[0])

                # Read vertices efficiently
                vertex_lines = [f.readline().strip() for _ in range(n_verts)]
                vertices = np.array([
                    [float(x) for x in line.split()[:3]]
                    for line in vertex_lines if line
                ], dtype=np.float32)

            # Sample or pad points if necessary
            if len(vertices) > self.num_points:
                indices = np.random.choice(len(vertices), self.num_points, replace=False)
                vertices = vertices[indices]
            elif len(vertices) < self.num_points:
                # Pad by repeating vertices with small noise
                repeats = self.num_points - len(vertices)
                indices = np.random.choice(len(vertices), repeats)
                noise = np.random.normal(0, 0.001, (repeats, 3)).astype(np.float32)
                vertices = np.vstack([vertices, vertices[indices] + noise])

            return vertices

        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            # Return a placeholder point cloud
            return np.zeros((self.num_points, 3), dtype=np.float32)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        point_cloud = self.read_off(file_path)

        # Center and normalize
        center = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - center
        furthest_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
        point_cloud = point_cloud / furthest_distance

        # Data augmentation for training
        if self.split == 'train':
            # Random rotation around z-axis
            theta = np.random.uniform(0, 2*np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            point_cloud = np.dot(point_cloud, rotation_matrix)

            # Small random jitter
            point_cloud += np.random.normal(0, 0.01, size=point_cloud.shape)

        # Apply custom transforms if provided
        if self.transform:
            point_cloud = self.transform(point_cloud)

        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)
        label = self.labels[idx]

        return point_cloud, label

    def get_category_name(self, label):
        """Get category name from label"""
        return self.categories[label]

    @staticmethod
    def get_category_mapping():
        """Return the mapping of category names to indices"""
        # This is the standard ModelNet40 category list
        categories = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
            'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
            'wardrobe', 'xbox'
        ]
        return {name: i for i, name in enumerate(categories)}
