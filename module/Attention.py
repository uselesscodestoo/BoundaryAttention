import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def unfold_neighborhood(x, neighborhood_size=11) -> torch.Tensor:
    """
    提取邻域，返回形状[B, H, W, N, N, C]
    x: 输入张量，形状为[B, H, W, C]
    neighborhood_size: 邻域大小（奇数）
    pad: 填充大小，确保边界像素有完整的邻域
    """
    B, H, W, C = x.shape
    N = neighborhood_size
    P = (N - 1) // 2
    if N % 2 == 0:
        raise ValueError("neighborhood_size must be an odd number")
    # 复制边界像素
    x_padded = F.pad(x, (0, 0, P, P, P, P))
    x_2d = x_padded.permute(0, 3, 1, 2)
    
    # 使用 unfold 提取邻域（2D）
    unfolded = F.unfold(x_2d, (N, N))
    unfolded = unfolded.view(B, C, N, N, H, W)
    # 调整形状为 [B, H, W, N, N, C]
    unfolded = unfolded.permute(0, 4, 5, 2, 3, 1)
    return unfolded


class RelativePositionalEncoding(nn.Module):
    """生成相对于邻域中心的位置编码（优化版）"""
    def __init__(self, neighbor_size, dim, device='cuda'):
        super().__init__()
        self.dim = dim
        self.neighbor_size = neighbor_size
        self.encoding = nn.Parameter(torch.randn(neighbor_size, neighbor_size, dim, device=device))
        self.device = device
    
    def forward(self):
        return self.encoding



class NeighborhoodCrossAttention(nn.Module):
    """优化版邻域交叉注意力组件（无显式循环）"""
    def __init__(self, dim, out_dim, neighborhood_size=11, num_heads=4, dropout=0.1, device='cuda'):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.neighborhood_size = neighborhood_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pad = neighborhood_size // 2
        self.device = device
        
        # 相对位置编码
        self.pos_encoding = RelativePositionalEncoding(
            neighbor_size=neighborhood_size,
            dim=dim,
            device=device
        )
        
        # MLP模块
        self.mlp = nn.Sequential(
            nn.Linear(neighborhood_size*neighborhood_size*num_heads, out_dim, device=device),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim, device=device)

        self.short_cut = nn.Linear(dim, out_dim, device=device) if dim!=out_dim else nn.Identity()
    
    def forward(self, q, kv):
        """
        前向传播（张量向量化实现）
        q: 查询张量，形状为[B, H, W, dim]
        kv: 键值张量，形状为[B, H, W, dim]
        """
        B, H, W, C = q.shape
        pad = self.pad
        N = self.neighborhood_size
        heads = self.num_heads
        head_dim = C // heads

        kv_neighbors = unfold_neighborhood(kv, N)

        # 生成相对位置编码
        pos_enc = self.pos_encoding()  # [N, N, C]
        
        # 整合位置编码到K
        kv_with_pos = kv_neighbors + pos_enc  # [B, H, W, N, N, C]

        q_mulit_head = q.reshape(B, H, W, heads, head_dim, 1)
        kv_mulit_head = kv_with_pos.reshape(B, H, W, heads, N**2, head_dim)
        
        # 计算注意力分数 [B, H, W, heads, N^2, head_dim] x [B, H, W, heads, head_dim, 1]
        attn_scores = torch.matmul(kv_mulit_head, q_mulit_head)
        attn_scores = attn_scores.reshape(B, H, W, -1)

        # 残差连接和层归一化
        mlp_output = self.mlp(attn_scores)
        q = self.norm(self.short_cut(q) + mlp_output)
        
        return q
    

if __name__ == "__main__":
    model = NeighborhoodCrossAttention(dim=64, out_dim=64, device='cpu')
    shape = (2, 32, 32, 64)
    q = torch.randn(shape)
    kv = torch.randn(shape)

    kv_neighbors = unfold_neighborhood(kv, 11)
    b = kv_neighbors[0,5,5]==kv[0,:11,:11,:]
    print(b.sum().item(), b.all().item())

    output = model(q, kv)
    print(output.shape)
