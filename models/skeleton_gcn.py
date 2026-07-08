"""
Skeleton Graph Convolutional Network for Pose Estimation

Utilizes human skeleton topology as structural prior.
Similar to DT-Pose's topology-constrained pose decoder.

Key features:
1. Skeleton adjacency matrix encoding joint connections
2. Graph convolution aggregates information from connected joints
3. Supports both 14-joint (WiFiPose3D) and 17-joint (MM-Fi) formats
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

SKELETON_14 = [
    (0, 1),
    (0, 2),
    (0, 3),
    (2, 4),
    (4, 8),
    (3, 6),
    (6, 10),
    (2, 5),
    (3, 7),
    (5, 7),
    (5, 9),
    (9, 12),
    (7, 11),
    (11, 13),
]

SKELETON_17 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

SKELETON_18 = [
    (4, 3), (3, 2),
    (7, 6), (6, 5),
    (10, 9), (9, 8),
    (13, 12), (12, 11),
    (8, 2), (11, 5),
    (2, 1), (5, 1),
    (0, 1),
    (14, 0), (15, 0),
    (16, 14), (17, 15),
]

def build_adjacency_matrix(num_joints: int, self_loop: bool = True) -> torch.Tensor:
    """
    Build skeleton adjacency matrix.

    Args:
        num_joints: 14 or 17
        self_loop: Whether to add self-connections

    Returns:
        A: [num_joints, num_joints] adjacency matrix (normalized)
    """
    skeleton = {14: SKELETON_14, 17: SKELETON_17, 18: SKELETON_18}[num_joints]

    A = torch.zeros(num_joints, num_joints)
    for (i, j) in skeleton:
        A[i, j] = 1
        A[j, i] = 1

    if self_loop:
        A = A + torch.eye(num_joints)

    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm

class GraphConvLayer(nn.Module):
    """
    Single Graph Convolutional Layer.

    X' = sigma(A * X * W)

    Where A is the normalized adjacency matrix.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_joints: int = 14,
                 bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        A = build_adjacency_matrix(num_joints, self_loop=True)
        self.register_buffer('A', A)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, in_features] joint features

        Returns:
            out: [B, J, out_features] updated joint features
        """

        support = torch.matmul(x, self.weight)
        output = torch.matmul(self.A, support)

        if self.bias is not None:
            output = output + self.bias

        return output

class SkeletonGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network for skeleton-based pose refinement.

    Takes joint features and refines them using skeleton topology.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int = 128,
                 out_features: int = 3,
                 num_layers: int = 3,
                 num_joints: int = 14,
                 dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.residual = residual
        self.num_joints = num_joints

        self.input_proj = nn.Linear(in_features, hidden_features)

        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.gcn_layers.append(
                GraphConvLayer(hidden_features, hidden_features, num_joints)
            )
            self.norms.append(nn.LayerNorm(hidden_features))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, out_features)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, in_features] joint features

        Returns:
            pose: [B, J, 3] refined pose coordinates
        """
        h = self.input_proj(x)

        for i in range(self.num_layers):
            h_new = self.gcn_layers[i](h)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            if self.residual:
                h = self.norms[i](h + h_new)
            else:
                h = self.norms[i](h_new)

        pose = self.output_proj(h)

        return pose

class GCNPoseDecoder(nn.Module):
    """
    GCN-based Pose Decoder combining Transformer and GCN.

    Similar to DT-Pose's topology-constrained decoder:
    - Transformer captures global joint relationships
    - GCN enforces local skeleton topology constraints

    Architecture:
        input features → Transformer → GCN → pose
    """

    def __init__(self,
                 in_features: int = 32,
                 hidden_dim: int = 128,
                 num_joints: int = 14,
                 num_transformer_layers: int = 2,
                 num_gcn_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.num_joints = num_joints

        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.joint_embed = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.gcn = SkeletonGCN(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=3,
            num_layers=num_gcn_layers,
            num_joints=num_joints,
            dropout=dropout,
            residual=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, in_features] joint features from PointTransformer

        Returns:
            pose: [B, J, 3] pose coordinates
        """
        B = x.shape[0]

        h = self.input_proj(x)

        h = h + self.joint_embed.expand(B, -1, -1)

        h = self.transformer(h)

        pose = self.gcn(h)

        return pose

class ChebyshevGraphConv(nn.Module):
    """
    Chebyshev Graph Convolution with K-hop aggregation.

    Reference: GraphPose-Fi (arXiv:2511.19105)

    Uses Chebyshev polynomials to approximate spectral graph convolutions:
    H' = sum_{k=0}^{K} T_k(L_tilde) @ H @ W_k

    Where:
    - L_tilde = 2/lambda_max * L - I (scaled Laplacian)
    - T_k are Chebyshev polynomials
    - This allows K-hop neighborhood aggregation without explicit stacking
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_joints: int = 17,
                 K: int = 3,
                 bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(K)
        ])

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        A = build_adjacency_matrix(num_joints, self_loop=True)
        L = self._build_laplacian(A)
        self.register_buffer('L', L)

        self._reset_parameters()

    def _build_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Build scaled Laplacian for Chebyshev approximation."""
        D = torch.diag(A.sum(dim=1))

        L = D - A

        D_inv_sqrt = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        lambda_max = 2.0
        L_tilde = (2.0 / lambda_max) * L_norm - torch.eye(A.shape[0])

        return L_tilde

    def _reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, in_features] joint features

        Returns:
            out: [B, J, out_features] updated joint features
        """
        B, J, F = x.shape

        T_0 = x
        T_1 = torch.matmul(self.L, x)

        output = torch.matmul(T_0, self.weights[0])
        if self.K > 1:
            output = output + torch.matmul(T_1, self.weights[1])

        T_km2, T_km1 = T_0, T_1
        for k in range(2, self.K):
            T_k = 2 * torch.matmul(self.L, T_km1) - T_km2
            output = output + torch.matmul(T_k, self.weights[k])
            T_km2, T_km1 = T_km1, T_k

        if self.bias is not None:
            output = output + self.bias

        return output

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-Head Graph Attention for adaptive neighbor weighting.

    Reference: GraphPose-Fi + GAT principles

    Unlike standard GCN which uses fixed adjacency weights,
    this learns attention weights between connected joints.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_joints: int = 17,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        self.q_proj = nn.Linear(in_features, out_features, bias=bias)
        self.k_proj = nn.Linear(in_features, out_features, bias=bias)
        self.v_proj = nn.Linear(in_features, out_features, bias=bias)
        self.out_proj = nn.Linear(out_features, out_features, bias=bias)

        A = build_adjacency_matrix(num_joints, self_loop=True)
        mask = (A > 0).float()
        self.register_buffer('adj_mask', mask)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, in_features] joint features

        Returns:
            out: [B, J, out_features] updated joint features
        """
        B, J, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, J, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, J, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, J, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        mask = self.adj_mask.unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, J, self.out_features)

        out = self.out_proj(out)

        return out

class ImprovedGCNBlock(nn.Module):
    """
    Improved GCN Block combining local (Chebyshev) and global (Self-Attention) processing.

    Architecture (per block):
        input → ChebyshevConv → LayerNorm → Self-Attention → LayerNorm → FFN → output
              └──────────────────────────────────────────────────────┘
                                    (residual)

    Reference: DT-Pose (local GCN + global Transformer) + GraphPose-Fi (Chebyshev + MHSA)
    """

    def __init__(self,
                 hidden_dim: int,
                 num_joints: int = 17,
                 K: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_graph_attention: bool = True):
        super().__init__()

        self.cheby_conv = ChebyshevGraphConv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_joints=num_joints,
            K=K
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        if use_graph_attention:
            self.attention = MultiHeadGraphAttention(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_joints=num_joints,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        self.use_graph_attention = use_graph_attention
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, hidden_dim] joint features

        Returns:
            out: [B, J, hidden_dim] updated joint features
        """
        h = self.cheby_conv(x)
        h = F.gelu(h)
        h = self.dropout(h)
        x = self.norm1(x + h)

        if self.use_graph_attention:
            h = self.attention(x)
        else:
            h, _ = self.attention(x, x, x)
        x = self.norm2(x + h)

        h = self.ffn(x)
        x = self.norm3(x + h)

        return x

class ImprovedSkeletonGCN(nn.Module):
    """
    Improved Skeleton GCN with Chebyshev convolutions and self-attention.

    Combines local (K-hop neighborhood) and global (joint-to-joint attention)
    processing for better pose estimation.

    Reference: DT-Pose + GraphPose-Fi architecture principles
    """

    def __init__(self,
                 in_features: int,
                 hidden_dim: int = 256,
                 out_features: int = 3,
                 num_joints: int = 17,
                 num_layers: int = 4,
                 K: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_graph_attention: bool = True):
        super().__init__()

        self.num_joints = num_joints

        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.joint_embed = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList([
            ImprovedGCNBlock(
                hidden_dim=hidden_dim,
                num_joints=num_joints,
                K=K,
                num_heads=num_heads,
                dropout=dropout,
                use_graph_attention=use_graph_attention
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, J, in_features] joint features

        Returns:
            pose: [B, J, 3] pose coordinates
        """
        B = x.shape[0]

        h = self.input_proj(x)

        h = h + self.joint_embed.expand(B, -1, -1)

        for block in self.blocks:
            h = block(h)

        pose = self.output_proj(h)

        return pose

if __name__ == "__main__":
    print("Testing SkeletonGCN...")

    gcn_14 = SkeletonGCN(in_features=32, num_joints=14)
    x_14 = torch.randn(4, 14, 32)
    out_14 = gcn_14(x_14)
    print(f"14-joint: input {x_14.shape} -> output {out_14.shape}")

    gcn_17 = SkeletonGCN(in_features=32, num_joints=17)
    x_17 = torch.randn(4, 17, 32)
    out_17 = gcn_17(x_17)
    print(f"17-joint: input {x_17.shape} -> output {out_17.shape}")

    print("\nTesting GCNPoseDecoder...")
    decoder = GCNPoseDecoder(in_features=32, num_joints=14)
    pose = decoder(x_14)
    print(f"GCNPoseDecoder: input {x_14.shape} -> pose {pose.shape}")

    params = sum(p.numel() for p in decoder.parameters())
    print(f"GCNPoseDecoder parameters: {params:,}")

    print("\n" + "=" * 50)
    print("Testing Improved GCN Modules...")
    print("=" * 50)

    print("\nTesting ChebyshevGraphConv...")
    cheby = ChebyshevGraphConv(in_features=64, out_features=64, num_joints=17, K=3)
    x = torch.randn(4, 17, 64)
    out = cheby(x)
    print(f"  ChebyshevConv: {x.shape} -> {out.shape}")
    params = sum(p.numel() for p in cheby.parameters())
    print(f"  Parameters: {params:,}")

    print("\nTesting MultiHeadGraphAttention...")
    mhga = MultiHeadGraphAttention(in_features=64, out_features=64, num_joints=17, num_heads=4)
    out = mhga(x)
    print(f"  MHGA: {x.shape} -> {out.shape}")
    params = sum(p.numel() for p in mhga.parameters())
    print(f"  Parameters: {params:,}")

    print("\nTesting ImprovedGCNBlock...")
    block = ImprovedGCNBlock(hidden_dim=64, num_joints=17, K=3, num_heads=4)
    out = block(x)
    print(f"  Block: {x.shape} -> {out.shape}")
    params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters: {params:,}")

    print("\nTesting ImprovedSkeletonGCN...")
    improved_gcn = ImprovedSkeletonGCN(
        in_features=256,
        hidden_dim=256,
        out_features=3,
        num_joints=17,
        num_layers=4,
        K=3,
        num_heads=4
    )
    x = torch.randn(4, 17, 256)
    pose = improved_gcn(x)
    print(f"  ImprovedGCN: {x.shape} -> {pose.shape}")
    params = sum(p.numel() for p in improved_gcn.parameters())
    print(f"  Parameters: {params:,}")

    print("\nAll tests passed!")
