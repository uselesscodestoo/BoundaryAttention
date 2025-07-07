import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import tqdm
import os
from torch.utils.data import Dataset, DataLoader


class ImageNPZDataset(Dataset):
    def __init__(self, image_dir, npz_dir, transform=None):
        """
        初始化图片与npz标签数据集
        
        参数:
            image_dir: 图片文件夹路径
            npz_dir: npz标签文件夹路径
            transform: 图片预处理转换
        """
        self.image_dir = image_dir
        self.npz_dir = npz_dir
        self.transform = transform
        
        # 获取图片文件名列表(假设图片和npz文件命名一致，仅扩展名不同)
        self.image_files = sorted(os.listdir(image_dir))
        
        # 过滤不支持的文件类型
        supported_img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = [f for f in self.image_files 
                           if os.path.splitext(f)[1].lower() in supported_img_extensions]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # 获取图片路径并加载
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 获取对应的npz文件路径并加载
        # 假设文件名相同，仅扩展名不同
        npz_name = os.path.splitext(img_name)[0] + '.npz'
        npz_path = os.path.join(self.npz_dir, npz_name)
        
        # 检查npz文件是否存在
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"对应的npz文件不存在: {npz_path}")
            
        # 加载npz数据(假设包含名为'label'的数组)
        npz_data = np.load(npz_path, allow_pickle=True)
        label = npz_data['label']  # 根据实际情况修改键名
        
        # 应用图片转换
        if self.transform:
            image = self.transform(image)
            
        # 将标签转换为张量
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
    
def create_dataset(image_dir, npz_dir, resize=None, batch_size=32, shuffle=True, num_workers=4):
    """
    创建数据集和数据加载器
    
    参数:
        image_dir: 图片文件夹路径
        npz_dir: npz标签文件夹路径
        resize: 图片调整大小
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
    """
    # 定义图片预处理转换
    if resize is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])  # ImageNet标准归一化
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])  # ImageNet标准归一化
        ])
    
    # 创建数据集
    dataset = ImageNPZDataset(
        image_dir=image_dir,
        npz_dir=npz_dir,
        transform=transform
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU数据传输
    )
    
    return dataset, dataloader

def load_dataset(data_dir,batch_size=8):
    img_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    
    dataset, dataloader = create_dataset(
        image_dir=img_dir,
        npz_dir=label_dir,
        batch_size=batch_size,
    )
    return dataset, dataloader

