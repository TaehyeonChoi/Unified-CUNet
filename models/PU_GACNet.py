import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionConvolution(nn.Module):
    """Graph Attention Convolution with fixed dimensions"""
    def __init__(self, in_channels, out_channels, k=9):
        super(GraphAttentionConvolution, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Feature transformation
        self.transform = nn.Conv1d(in_channels, out_channels, 1)

        # Attention network
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, 1),
            nn.Softmax(dim=-1)
        )

    def get_graph_feature(self, x):
        # x: [B, C, N]
        batch_size, num_dims, num_points = x.size()
        k = min(self.k, num_points)

        x_trans = x.transpose(2, 1).contiguous()  # [B, N, C]

        # Compute pairwise distances
        inner = -2 * torch.matmul(x_trans, x)  # [B, N, N]
        xx = torch.sum(x_trans ** 2, dim=2, keepdim=True)  # [B, N, 1]
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B, N, N]

        # Get k nearest neighbors
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]

        # Get feature from k neighbors
        feature_list = []
        for bi in range(batch_size):
            feature = x[bi].unsqueeze(0)  # [1, C, N]
            neighbor_features = torch.index_select(
                feature, 2, idx[bi].view(-1)
            ).view(1, num_dims, num_points, k)  # [1, C, N, k]
            feature_list.append(neighbor_features)

        # Stack features from all batches
        neighbor_features = torch.cat(feature_list, dim=0)  # [B, C, N, k]

        # Prepare central features
        central_feature = x.unsqueeze(-1).repeat(1, 1, 1, k)  # [B, C, N, k]

        # Combine features
        edge_feature = torch.cat([central_feature, neighbor_features - central_feature], dim=1)  # [B, 2C, N, k]

        return edge_feature

    def forward(self, x):
        # x: [B, C, N]
        batch_size, num_dims, num_points = x.size()
        k = min(self.k, num_points)

        # Get edge features
        edge_feature = self.get_graph_feature(x)  # [B, 2C, N, k]

        # Compute attention weights
        attention_weights = self.attention(edge_feature)  # [B, 1, N, k]

        # Transform features
        transformed = self.transform(x)  # [B, out_C, N]
        transformed = transformed.unsqueeze(-1).repeat(1, 1, 1, k)  # [B, out_C, N, k]

        # Apply attention and aggregate
        out = (transformed * attention_weights).sum(dim=-1)  # [B, out_C, N]

        return out

class EdgeAwareNodeShuffle(nn.Module):
    """Edge-aware NodeShuffle with fixed dimensions"""
    def __init__(self, in_channels, up_factor=4):
        super(EdgeAwareNodeShuffle, self).__init__()
        self.up_factor = up_factor

        self.edge_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * up_factor, 1),
            nn.BatchNorm1d(in_channels * up_factor),
            nn.ReLU()
        )

    def forward(self, x, edge_info):
        # x, edge_info: [B, C, N]
        batch_size, channels, num_points = x.size()

        # Process edge features
        combined = torch.cat([x, edge_info], dim=1)  # [B, 2C, N]
        edge_features = self.edge_conv(combined)  # [B, C, N]

        # Expand features
        expanded = self.expand_conv(edge_features)  # [B, C*up_factor, N]
        output = expanded.view(batch_size, channels, self.up_factor, num_points)
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, channels, -1)  # [B, C, N*up_factor]

        return output

class PUGACNet(nn.Module):
    """Point Cloud Upsampling Network with Graph Attention"""
    def __init__(self, in_channels=3, out_channels=3, up_factor=4):
        super(PUGACNet, self).__init__()
        self.up_factor = up_factor

        # Feature extraction
        self.gac1 = nn.Sequential(
            GraphAttentionConvolution(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.gac2 = nn.Sequential(
            GraphAttentionConvolution(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Feature expansion
        self.ens = EdgeAwareNodeShuffle(128, up_factor)

        # Coordinate reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, 1)
        )

    def forward(self, x):
        # Input shape verification
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 input channels, got {x.size(1)}")

        # Feature extraction
        f1 = self.gac1(x)
        f2 = self.gac2(f1)

        # Edge information
        edge_info = f2.detach()

        # Upsampling
        expanded = self.ens(f2, edge_info)

        # Reconstruction
        output = self.reconstruction(expanded)

        return output
