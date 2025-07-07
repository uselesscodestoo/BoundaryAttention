import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import tqdm
from dataset import load_dataset


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FCNResidual(nn.Module):
    def __init__(self, num_classes=7):
        super(FCNResidual, self).__init__()
        
        # 编码器部分 - 特征提取
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256)
        )
        
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512)
        )
        
        # 桥接层
        self.bridge = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 1024)
        )
        
        # 解码器部分 - 上采样
        self.decoder4 = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )
        
        self.decoder3 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        self.decoder2 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        
        self.decoder1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        # 最终分类层
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # 跳跃连接的上采样层
        self.upsample_bridge = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器路径，保存中间特征用于跳跃连接
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # 桥接层
        bridge = self.bridge(enc4)
        
        # 解码器路径，使用跳跃连接
        dec4 = self.decoder4(self.upsample_bridge(bridge) + enc4)
        dec3 = self.decoder3(self.upsample4(dec4) + enc3)
        dec2 = self.decoder2(self.upsample3(dec3) + enc2)
        dec1 = self.decoder1(self.upsample2(dec2) + enc1)
        
        # 最终分类
        out = self.final(dec1)
        
        return out

# 测试网络
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNResidual(num_classes=7)
    model = model.to(device)

    dataset, dataloader = load_dataset('./synthetic_dataset/', batch_size=8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm.tqdm(range(10)):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        

    torch.save(model.state_dict(), 'last.pth')
