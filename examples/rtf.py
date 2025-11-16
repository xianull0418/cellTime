import torch
import torch.nn.functional as F

# Rectified Flow可以简单的理解成从A点到B点，也就是要有一个路线（流）、一个速度（流速）、针对每个点的采样


class RectifiedFlow:
    # 定义轨迹的点（欧拉采样）
    def euler(self, x_t, v, dt):
        """使用欧拉方法采样：ODE f(t+dt) = f(t) + f'(t) * dt

        Args:
            x_t (tensor): 当前时刻的值
            v (tensor): 速度
            dt (float): 时间步长

        Returns:
            tensor: 下一个时刻的值
        """
        x_t = x_t + v*dt

        return x_t

    # 路线
    def create_flow(self, x_1, t):
        """使用rtf核心公式x_t = t * x_1 + (1 - t) * x_0构建x_0 -> x_1的路线，在起点 x_0 （随机噪声）和终点 x_1 （目标数据）之间创建线性插值路径。

        Args:
            x_1 (tensor): 原始图像，维度为[B,C,H,W]
            t (tensor): 时间步长，维度为[B]，重复三次 None 操作，最终将形状从 [B] 扩展为 [B, 1, 1, 1]

        Returns:
            tensor: x_t, x_0
        """
        x_0 = torch.randn_like(x_1)

        t = t[:, None, None, None] # [B, 1, 1, 1]，便于广播

        x_t = t * x_1 + (1 - t) * x_0

        return x_t, x_0

    # 损失函数
    def mse_loss(self, v, x_1, x_0):
        """计算RTF的损失函数：MSE(x_1 - x_0, v)

        Args:
            v (tensor): 速度，维度为[B,C,H,W]
            x_1 (tensor): 原始图像，维度为[B,C,H,W]
            x_0 (tensor): 随机噪声，维度为[B,C,H,W]

        Returns:
            float: 损失值
        """
        loss = F.mse_loss(x_1 - x_0, v)
        # loss = torch.mean((x_1 - x_0- v）**2)

        return loss


if __name__ == "__main__":
    # 时间越大，越接近原始图像

    rf = RectifiedFlow()
    t = torch.full((2,), 0.999)
    x_t, x_0 = rf.create_flow(torch.zeros(2,3,4,4), t)
    

    print(x_t)
    print(x_0)