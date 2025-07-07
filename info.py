import torch
import torch.nn as nn

def estimate_inference_memory(model, input_shape, device='cuda'):
    # 把模型移到指定设备
    model = model.to(device)
    # 开启评估模式
    model.eval()
    
    # 记录初始内存使用情况
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device)
    
    # 创建输入张量
    input_tensor = torch.randn(*input_shape, device=device)
    
    # 进行一次前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    # 计算内存使用量
    end_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    print(f'前向传播结束内存占用: {end_mem / 1024**2:.2f} MB')
    print(f'峰值内存占用: {peak_mem / 1024**2:.2f} MB')
    
    # 计算模型参数占用的内存
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # 计算前向传播过程中的内存使用量（不包含参数）
    forward_mem = peak_mem - start_mem - param_size
    
    return {
        'model_params_size': param_size,
        'forward_pass_memory': forward_mem,
        'total_inference_memory': peak_mem - start_mem
    }

def estimate_training_memory(model, input_shape, optimizer_class=torch.optim.Adam, device='cuda'):
    # 把模型移到指定设备
    model = model.to(device)
    # 开启训练模式
    model.train()
    
    # 记录初始内存使用情况
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device)
    
    # 创建输入张量和目标张量
    input_tensor = torch.randn(*input_shape, device=device)
    
    # 创建优化器
    optimizer = optimizer_class(model.parameters(), lr=0.001)
    
    # 执行一次前向传播
    output = model(input_tensor)
    # 计算损失
    target = torch.randn_like(output, device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # 记录前向传播后的内存
    forward_mem = torch.cuda.memory_allocated(device) - start_mem
    
    # 执行反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 记录反向传播后的内存
    backward_mem = torch.cuda.memory_allocated(device) - start_mem
    
    # 计算优化器状态占用的内存
    optimizer.step()
    optimizer_mem = torch.cuda.memory_allocated(device) - start_mem - backward_mem
    
    # 获取峰值内存使用量
    peak_mem = torch.cuda.max_memory_allocated(device) - start_mem
    
    return {
        'model_params_size': sum(p.numel() * p.element_size() for p in model.parameters()),
        'forward_pass_memory': forward_mem,
        'backward_pass_memory': backward_mem - forward_mem,
        'optimizer_state_memory': optimizer_mem,
        'total_training_memory': peak_mem
    }

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    计算模型的总参数量和可训练参数量
    
    参数:
        model: PyTorch模型
    
    返回:
        total_params: 总参数量
        trainable_params: 可训练参数量（requires_grad=True的参数）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    from module.Boundary import BoundaryAttentionModule
    model = BoundaryAttentionModule(3, 64, 8)
    print(count_parameters(model))