
import torch
from torch import nn
from torch.nn import *


class CS_Branch(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False, weight = 0.02):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        # 卷积层
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)

        self.FC = nn.Sequential(
            Linear(in_features=out_channels, out_features=out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(in_features=out_channels, out_features=out_channels),
            Sigmoid()
        )
        self.flatten = Flatten()
        self.average_pool_odd = AvgPool1d(kernel_size=2, stride=2, padding=1)
        self.average_pool_even = AvgPool1d(kernel_size=2, stride=2, padding=0)

        for n, m in self.named_modules():
            if isinstance(m, Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
        self.dropout = nn.Dropout(0.1)
        self.threshold_scale = nn.Parameter(torch.tensor(weight))

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.dropout(self.FC(gap))
        # 计算阈值并进行软阈值化
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)
        x = self.threshold_scale * x

        if self.down_sample:
            if input.size(2) % 2 == 1:
                input = self.average_pool_odd(input)
            else :
                input = self.average_pool_even(input)

        if self.in_channels != self.out_channels:
            input = nn.functional.interpolate(input.unsqueeze(2), size=(self.out_channels, input.size(2)), mode='nearest').squeeze(2)
            input = input[:, 0, :, :]
        result = input + x
        return result
