import torch
import torch.nn as nn
import torch.nn.functional as F

class DOAResNet(nn.Module):
    def __init__(self, number_element, output_size):
        super(DOAResNet, self).__init__()
        self.inchannel = 64
        self.M = number_element
        # f_1 卷积层
        self.conv1 = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=(2*self.M,3), stride=(2*self.M,1), padding=(0, 1), bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
        )
        # f_2 池化层
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2))

        # f_3 ~ f_6 残差块
        self.layer1 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 512, 1, stride=2)
        self.layer3= self.make_layer(ResidualBlock, 1024, 1, stride=2)

        # f_8 全连接层
        fc = []
        fc.append(nn.Linear(1024, 512))
        fc.append(nn.ReLU())
        # fc.append(nn.Dropout(0.2))
        fc.append(nn.Linear(512, output_size))
        self.fc = nn.Sequential(*fc)
        
    # 此函数主要是用来重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 将复数信号处理为单通道实数信号 [64, 12, 10]
        bts, M, L = x.shape
        x = torch.view_as_real(x)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(bts, M * 2, L) 
        x = x.unsqueeze(1)

        # f_1 卷积层
        x = self.conv1(x)  # [btz, 64, 1, 10]
        # f_2 池化层
        x = self.maxpool(x)  # [btz, 64, 1, 4]

        # f_3 ~ f_6 残差块
        x = self.layer1(x) # [btz, 64,  1, 4]
        x = self.layer2(x) # [btz, 512, 1, 2]
        x = self.layer3(x) # [btz, 1024, 1, 1]
        
        # f_7 全局平均池化层
        x = F.avg_pool2d(x, x.size()[2:])  # [btz, 1024, 1, 1]
        
        # f_8 全连接层
        x = x.view(x.size(0), -1)  # [btz, 1024]
        x = self.fc(x)  # [btz, 1024] - [btz, 512] - [btz, 121]
        
        return x

    

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self,
                 inchannel,
                 outchannel, 
                 stride=1,
                 ):
        super(ResidualBlock, self).__init__()
        # 残差块的左侧
        self.left = nn.Sequential(
          nn.Conv2d(inchannel, outchannel, kernel_size=(1,3), stride=(1,stride), padding=(0,1), bias=False),
          nn.BatchNorm2d(outchannel),
          nn.ReLU(inplace=True),
          nn.Conv2d(outchannel, outchannel, kernel_size=(1,3), stride=1, padding=(0,1), bias=False),
          nn.BatchNorm2d(outchannel)
        )
        # 残差块的右侧
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
              nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=(1,stride), bias=False),
              nn.BatchNorm2d(outchannel)
            )
    
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out