import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborhoodMixingBlock(nn.Module):
    """
    邻域混合块，包含空间补丁混合器和通道混合器
    """
    def __init__(self, embed_dim, device='cuda'):
        """
        初始化邻域混合块
        
        参数:
            embed_dim: 嵌入维度
            hidden_dim: 通道混合器隐藏维度
        """
        super(NeighborhoodMixingBlock, self).__init__()
        
        # 空间补丁混合器：两个3x3卷积，跨通道共享权重
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, device=device),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, device=device),
            nn.GELU()
        )
        
        # 通道混合器：每个像素的MLP，空间上共享权重
        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, device=device),
            nn.GELU(),
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图，形状为(batch_size, height, width, embed_dim)
            
        返回:
            输出特征图，形状为(batch_size, height, width, embed_dim)
        """
        batch_size, height, width, embed_dim = x.shape
        
        # 调整输入形状以应用2D卷积
        # [batch, height, width, embed_dim] -> [batch, embed_dim, height, width]
        x_conv = x.permute(0, 3, 1, 2)
        
        # 空间补丁混合器
        x_spatial = self.spatial_mixer(x_conv)
        # 恢复原始形状
        x_spatial = x_spatial.permute(0, 2, 3, 1)
        
        # 残差连接
        x = x + x_spatial
        
        # 通道混合器：调整形状以应用MLP
        # [batch, height, width, embed_dim] -> [batch*height*width, embed_dim]
        x_channel = self.channel_mixer(x)
        
        # 第二个残差连接
        x = x + x_channel
        
        return x
    


class NeighborhoodMLPMixer(nn.Module):
    """
    邻域MLP-Mixer模块实现，基于论文"S3.1 Neighborhood MLP-Mixer"描述
    """
    def __init__(self, in_channels=3, embed_dim=64, hidden_dim=None,device='cuda'):
        """
        初始化邻域MLP-Mixer模块
        
        参数:
            in_channels: 输入通道数，默认为3(RGB)
            embed_dim: 嵌入维度，论文中使用64
            hidden_dim: 通道混合器隐藏维度，默认为embed_dim的4倍
        """
        super(NeighborhoodMLPMixer, self).__init__()
        
        # 输入投影层：将输入从in_channels投影到embed_dim
        self.input_projection = nn.Linear(in_channels, embed_dim, device=device)

        
        # 隐藏维度默认为嵌入维度的4倍
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        
        # 两个邻域混合块
        self.block1 = NeighborhoodMixingBlock(embed_dim, device=device)
        self.block2 = NeighborhoodMixingBlock(embed_dim, device=device)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量，形状为(batch_size, height, width, channels)
            
        返回:
            输出特征图，形状为(batch_size, height, width, embed_dim)
        """
        batch_size, height, width, channels = x.shape
        
        # [batch, height, width, channels] -> [batch*height*width, channels]
        x_proj = self.input_projection(x)
        
        # 两次邻域混合块
        x_block1 = self.block1(x_proj)
        x_block2 = self.block2(x_block1)
        
        # 裁剪回原始尺寸以移除填充的边界
        # 从填充后的形状(batch, h+2, w+2, embed_dim)裁剪为(batch, h, w, embed_dim)
        output = x_block2
        
        return output


# 示例用法
if __name__ == "__main__":
    # 创建一个随机输入张量 (batch_size=2, height=32, width=32, channels=3)
    input_tensor = torch.randn(2, 32, 32, 3, device='cuda')
    
    # 初始化NeighborhoodMLPMixer
    mixer = NeighborhoodMLPMixer(in_channels=3, embed_dim=64)
    
    # 前向传播
    output = mixer(input_tensor)
    
    # 打印输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")