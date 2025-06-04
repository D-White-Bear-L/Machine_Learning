import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    """
    Inception模块是GooLeNet的核心组件，通过并行使用不同大小的卷积核来提取多尺度特征
    包含四个分支：
    1. 1x1卷积
    2. 1x1卷积接3x3卷积
    3. 1x1卷积接5x5卷积 
    4. 3x3最大池化接1x1卷积
    最终将四个分支的输出在通道维度上拼接
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1分支 - 直接使用1x1卷积提取特征
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        
        # 1x1 -> 3x3分支 - 先用1x1降维再3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),  # 降维
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 3x3卷积
        )
        
        # 1x1 -> 5x5分支 - 先用1x1降维再5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),  # 降维
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 5x5卷积
        )
        
        # 3x3池化 -> 1x1分支 - 最大池化后接1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 保持尺寸不变的最大池化
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)  # 1x1卷积
        )
    
    def forward(self, x):
        # 并行计算四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 在通道维度(channel=1)上拼接四个分支的输出
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class GooLeNet(nn.Module):
    """
    GooLeNet (Inception-v1)网络结构
    主要特点：
    - 使用Inception模块构建深层网络
    - 通过1x1卷积降维减少计算量
    - 使用全局平均池化替代全连接层减少参数
    - 包含辅助分类器(本实现中省略)
    """
    def __init__(self, num_classes=1000):
        super(GooLeNet, self).__init__()
        
        # 初始卷积层组 --------------------------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 输入图像处理
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 下采样
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)                      # 1x1卷积
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)          # 3x3卷积扩展通道
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 下采样
        
        # Inception模块组 3a-3b -----------------------------------------
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)   # 第一个Inception模块
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64) # 第二个Inception模块
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 下采样
        
        # Inception模块组 4a-4e -----------------------------------------
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 下采样
        
        # Inception模块组 5a-5b -----------------------------------------
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # 最终分类层 ----------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化替代全连接层
        self.dropout = nn.Dropout(0.4)               # 防止过拟合
        self.fc = nn.Linear(1024, num_classes)       # 最后的分类层
    
    def forward(self, x):
        # 前向传播过程
        # 初始卷积层组
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        
        # 第一组Inception模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        # 第二组Inception模块
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        # 第三组Inception模块
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # 分类层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        return x

def count_parameters(model):
    """
    计算模型的可训练参数总数
    返回:
        int: 模型中所有requires_grad=True的参数的个数总和
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 创建模型并计算参数
model = GooLeNet()
total_params = count_parameters(model)
print(f"总可训练参数数量: {total_params:,}")