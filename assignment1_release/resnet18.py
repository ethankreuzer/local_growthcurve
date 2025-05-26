'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        #Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        out = F.relu(self.bn1(self.conv1(x)))  # First conv + bn + relu
        out = self.bn2(self.conv2(out))  # Second conv + bn
        out += self.shortcut(x)  # Add shortcut connection
        out = F.relu(out)  # Apply final ReLU
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                       stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(?)
        #self.layer1 = self._make_layer(64, 64, stride=1)
        #self.layer2 = self._make_layer(?, 128, stride=2)
        #self.layer3 = self._make_layer(?, 256, stride=2)
        #self.layer4 = self._make_layer(?, 512, stride=2)
        #self.linear = nn.Linear(?, num_classes)


        # Initial convolution layer (3x3 kernel, 64 channels, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        
        x = F.relu(self.bn1(self.conv1(images)))  # Initial conv + bn + relu
        x = self.layer1(x)  # First stage (64 channels)
        x = self.layer2(x)  # Second stage (128 channels)
        x = self.layer3(x)  # Third stage (256 channels)
        x = self.layer4(x)  # Fourth stage (512 channels)

        x = self.avgpool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten for FC layer
        logits = self.linear(x)  # Fully connected layer
        return logits

    def visualize(self, logdir="/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_5"):
        """ Visualize the kernel in the desired directory """
        '''
        kernels = self.conv1.weight.data.clone()  # Shape: [64, 3, 3, 3]

        # Normalize for visualization
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

        # Save the kernel visualization as a grid
        grid = vutils.make_grid(kernels, nrow=8, normalize=True, scale_each=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # Convert to numpy for display
        plt.axis("off")
        
        # Save the image
        save_path = os.path.join(logdir, "conv1_kernels.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Kernel visualization saved to {save_path}")
        '''
        os.makedirs(logdir, exist_ok=True)

        filters = self.conv1.weight.data.clone()

        
        filters = (filters - filters.min()) / (filters.max() - filters.min())

        grid = vutils.make_grid(filters, nrow=8, normalize=True, padding=2)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")

        filepath = os.path.join(logdir, "resnet18_filters.png")
        plt.savefig(filepath)
        plt.close()
