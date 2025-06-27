import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 加入 math 库用于 cos/sin

class RotatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, angle_deg=45):
        super().__init__()
        self.angle_rad = math.radians(angle_deg)  # 用 math 计算弧度
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)#保证输出和输入尺寸相同

    def get_rotation_grid(self, x, angle_rad, reverse=False):
        """
        生成旋转 grid：如果 reverse=True，表示旋转回来
        """
        theta = -angle_rad if reverse else angle_rad
        B, C, H, W = x.size()

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        theta_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0]
        ], dtype=torch.float, device=x.device).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta_matrix, size=x.size(), align_corners=False)
        return grid

    def rotate(self, x, reverse=False):
        grid = self.get_rotation_grid(x, self.angle_rad, reverse)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='zeros')

    def forward(self, x):
        # Step 1: 旋转图像
        x_rotated = self.rotate(x, reverse=False)

        # Step 2: 卷积操作（在旋转后的图像上提取特征）
        y = self.conv(x_rotated)

        # Step 3: 将输出旋转回来
        y_rotated_back = self.rotate(y, reverse=True)

        return y_rotated_back

if __name__ == '__main__':
    model = RotatedConv2D(3, 64, 3, 5)  # 5 度旋转
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
