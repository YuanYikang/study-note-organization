import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

# 定义U-Net模型组件

class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码（用于表示序列位置信息的常用方法）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
        return embeddings

class Block(nn.Module):
    """U-Net中的残差块"""
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        # 嵌入时间步信息
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t=None):
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        if self.time_mlp and t is not None:
            time_emb = self.relu(self.time_mlp(t))
            time_emb = time_emb[(..., ) + (None, ) * 2]  # 扩展维度以匹配h
            h = h + time_emb
            
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class UNet(nn.Module):
    """用于DDPM的U-Net架构"""
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        
        # 时间步编码
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 下采样部分
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()),
            Block(64, 128, time_dim),
            Block(128, 256, time_dim),
            Block(256, 256, time_dim),
        ])
        
        # 中间部分
        self.mid = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 上采样部分
        self.ups = nn.ModuleList([
            Block(256, 256, time_dim, up=True),
            Block(256+256, 128, time_dim, up=True),
            Block(128+128, 64, time_dim, up=True),
            nn.Sequential(
                nn.Conv2d(64+64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, 3, padding=1)
            )
        ])

    def forward(self, x, timestep):
        # 时间编码
        t = self.time_mlp(timestep)
        
        # 存储下采样特征用于跳跃连接
        residual_inputs = []
        
        # 下采样
        for down in self.downs[:-1]:
            if isinstance(down, Block):
                x = down(x, t)
            else:
                x = down(x)
            residual_inputs.append(x)
        
        # 最后一个下采样块不需要存储
        x = self.downs[-1](x, t)
        
        # 中间处理
        x = self.mid(x)
        
        # 上采样与跳跃连接
        for up, residual in zip(self.ups[:-1], reversed(residual_inputs)):
            if isinstance(up, Block):
                x = torch.cat([x, residual], dim=1)
                x = up(x, t)
            else:
                x = up(x)
        
        # 最后一层
        x = torch.cat([x, residual_inputs[0]], dim=1)
        x = self.ups[-1](x)
        
        return x

# 定义DDPM模型
class DDPM(nn.Module):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=28):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        # 定义噪声调度
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # 定义模型
        self.model = UNet().to(device)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, n):
        """采样生成新图像"""
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)
            for i in tqdm(reversed(range(1, self.noise_steps)), desc='采样中...'):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                predicted_noise = self.model(x, t)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                # 根据DDPM论文中的公式计算x_{t-1}
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x
    
    def forward(self, x, t):
        return self.model(x, t)

# 定义训练函数
def train_ddpm(model, dataloader, optimizer, epochs=30):
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # 选择随机时间步
            t = model.sample_timesteps(batch_size).to(device)
            
            # 添加噪声
            x_t, noise = model.noise_images(images, t)
            
            # 使用模型预测噪声
            predicted_noise = model(x_t, t)
            
            # 计算损失
            loss = F.mse_loss(predicted_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
        # 每个epoch结束后生成一些示例
        if (epoch+1) % 5 == 0:
            x_generated = model.sample(10)
            
            # 保存生成的图像
            plt.figure(figsize=(10, 1))
            for i in range(10):
                plt.subplot(1, 10, i+1)
                plt.imshow(x_generated[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'ddpm_samples_epoch_{epoch+1}.png')
            plt.close()
    
    return model

# 初始化和训练DDPM模型
ddpm = DDPM()
optimizer = optim.Adam(ddpm.parameters(), lr=1e-4)

# 训练模型
train_ddpm(ddpm, train_loader, optimizer, epochs=30)

# 生成最终样本
with torch.no_grad():
    x_generated = ddpm.sample(20)
    
    plt.figure(figsize=(10, 2))
    for i in range(20):
        plt.subplot(2, 10, i+1)
        plt.imshow(x_generated[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('ddpm_final_samples.png')
    plt.show()

# 保存模型
torch.save(ddpm.state_dict(), 'ddpm_mnist.pt')