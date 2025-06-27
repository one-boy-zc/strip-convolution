import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 加入 math 库用于 cos/sin


class StripConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k1, k2, groups=1, **kwargs):
        """
        条带方向 + 深度/分组卷积模块
        参数：
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - k1, k2: 条带方向卷积核尺寸（例如 k1=1, k2=11 表示水平）
        - groups: 分组数（默认 1 表示普通卷积，=in_channels 表示深度卷积）
        - kwargs: 其他 nn.Conv2d 支持的参数（如 stride, padding, dilation 等）
        """
        super().__init__()
        # 条带卷积1（例如 1×11）
        self.conv_spatial1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(k1, k2),
            padding=(k1 // 2, k2 // 2),
            groups=groups,
            **kwargs
        )

        # 条带卷积2（例如 11×1）
        self.conv_spatial2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(k2, k1),
            padding=(k2 // 2, k1 // 2),
            groups=groups,
            **kwargs
        )

    def forward(self, x):
        x = self.conv_spatial1(x)
        x = self.conv_spatial2(x)
        return x


if __name__ == '__main__':
    model = StripConv2d(3, 64, 3,5)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)