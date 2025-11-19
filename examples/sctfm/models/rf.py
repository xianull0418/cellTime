import torch
from torch import nn


class RFImage(nn.Module):
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,), device=x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,), device=x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b, device=z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b, device=z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

class RFSet_v1(nn.Module): # 这个版本想直接从状态1扩散到状态2，但是这样的做法应该不对，需要再看看论文
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln

    def forward(self, x1, x2, t1, t2, cond):
        b = x1.size(0)
        if self.ln:
            nt = torch.randn((b,), device=x1.device)
            rand_t = torch.sigmoid(nt)
        else:
            rand_t = torch.rand((b,), device=x1.device)

        texp = rand_t.view([b, *([1] * len(x1.shape[1:]))])
        # z1 = torch.randn_like(x)
        zt = (1 - texp) * x1 + texp * x2
        dt = (1 - texp) * t1[:,None] + texp * t2[:,None]
        vtheta = self.model(zt, dt, cond)
        # print('t1.shape, t2.shape', t1.shape, t2.shape)
        # print('texp.shape, dt.shape', texp.shape)
        # print('zt.shape, dt.shape', zt.shape, dt.shape)
        # print('vtheta.shape', vtheta.shape)
        # TODO 时序临时处理 t2-t1，待改进
        batchwise_mse = (((x2 - x1)/((t2-t1)[:,None]) - vtheta) ** 2).mean(dim=list(range(1, len(x1.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(rand_t, tlist)]
        return batchwise_mse.mean(), ttloss

    # TODO 实现采样
    @torch.no_grad()
    def sample(self, x1, t1, t2, cond, null_cond=None, sample_steps=50, cfg=2.0):
        z = x1
        b = z.size(0)
        dt = t2 - t1
        t_step = 1.0 / sample_steps
        # t_step = torch.tensor([t_step] * b, device=z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps):
            t_exp = i / sample_steps
            t_exp = torch.tensor([t_exp] * b, device=z.device)
            t = dt*t_exp + t1
            vc = self.model(z, t, cond)
            # print('t1, t2, t', t1, t2, t)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vu - vc)
            # print('z.shape, t_step.shape, dt.shape, vc.shape', z.shape, t_step.shape, dt.shape, vc.shape)
            z = z + t_step * dt[:,None] * vc
            images.append(z)
        return images


class RFSet(nn.Module):
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln

    def forward(self, x1, x2, t1, t2, cond):
        # b = x.size(0)
        # if self.ln:
        #     nt = torch.randn((b,), device=x.device)
        #     t = torch.sigmoid(nt)
        # else:
        #     t = torch.rand((b,), device=x.device)
        # texp = t.view([b, *([1] * len(x.shape[1:]))])
        # z1 = torch.randn_like(x)
        # zt = (1 - texp) * x + texp * z1
        # vtheta = self.model(zt, t, cond)
        # batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        # tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        # ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        # return batchwise_mse.mean(), ttloss
        #
        # print('x1.shape, x2.shape, t1.shape, t2.shape', x1.shape, x2.shape, t1.shape, t2.shape)
        b = x1.size(0)
        if self.ln:
            nt = torch.randn((b,), device=x1.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,), device=x1.device)

        texp = t.view([b, *([1] * len(x1.shape[1:]))])
        # print(texp.shape)
        # z1 = torch.randn_like(x1)
        # zt1 = (1 - texp) * x1 + texp * z1
        # z2 = torch.randn_like(x2)
        # zt2 = (1 - texp) * x2 + texp * z2
        t2 = torch.cat([t, t], dim=0)
        x = torch.cat([x1, x2],dim=0)
        x = x.clip(0,5) # TODO 临时
        texp2 = torch.cat([texp, texp], dim=0)
        z = torch.randn_like(x)
        zt = (1 - texp2) * x + texp2 * z
        # z = torch.cat([z1, z2], dim=0) # 噪声
        # zt = torch.cat([zt1, zt2], dim=0) # 中间状态
        cond2 = cond
        if cond2 is not None:
            cond2 = torch.cat([cond, cond], dim=0)
#         print('zt.shape, t.shape', zt.shape, t.shape)
        vtheta = self.model(zt, t2, cond2)
        batchwise_mse = ((z - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))


        # vtheta = self.model(zt1, rand_t, cond)
        # vtheta = self.model(zt2, rand_t, cond)

        # zt = (1 - texp) * x1 + texp * x2
        # dt = (1 - texp) * t1[:,None] + texp * t2[:,None]
        # vtheta = self.model(zt, dt, cond)
        # print('t1.shape, t2.shape', t1.shape, t2.shape)
        # print('texp.shape, dt.shape', texp.shape)
        # print('zt.shape, dt.shape', zt.shape, dt.shape)
        # print('vtheta.shape', vtheta.shape)
        # TODO 时序临时处理 t2-t1，待改进
        # batchwise_mse = (((x2 - x1)/((t2-t1)[:,None]) - vtheta) ** 2).mean(dim=list(range(1, len(x1.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t2, tlist)]
        return batchwise_mse.mean(), ttloss

    # TODO 实现采样
    @torch.no_grad()
    def sample(self, x1, t1, t2, cond, null_cond=None, sample_steps=50, cfg=2.0):
        z = x1
        b = z.size(0)
        dt = t2 - t1
        t_step = 1.0 / sample_steps
        # t_step = torch.tensor([t_step] * b, device=z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        # for i in range(sample_steps):
        #     t_exp = i / sample_steps
        #     t_exp = torch.tensor([t_exp] * b, device=z.device)
        #     t = dt*t_exp + t1
        #     vc = self.model(z, t, cond)
        #     # print('t1, t2, t', t1, t2, t)
        #     if null_cond is not None:
        #         vu = self.model(z, t, null_cond)
        #         vc = vu + cfg * (vu - vc)
        #     # print('z.shape, t_step.shape, dt.shape, vc.shape', z.shape, t_step.shape, dt.shape, vc.shape)
        #     z = z - t_step * dt[:,None] * vc
        #     images.append(z)
        return images

class RFTSet(nn.Module):
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln

    def forward(self, x1, x2, t1, t2, cond):
        # b = x.size(0)
        # if self.ln:
        #     nt = torch.randn((b,), device=x.device)
        #     t = torch.sigmoid(nt)
        # else:
        #     t = torch.rand((b,), device=x.device)
        # texp = t.view([b, *([1] * len(x.shape[1:]))])
        # z1 = torch.randn_like(x)
        # zt = (1 - texp) * x + texp * z1
        # vtheta = self.model(zt, t, cond)
        # batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        # tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        # ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        # return batchwise_mse.mean(), ttloss
        #
        # print('x1.shape, x2.shape, t1.shape, t2.shape', x1.shape, x2.shape, t1.shape, t2.shape)
        b = x1.size(0)
        if self.ln:
            nt = torch.randn((b,), device=x1.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,), device=x1.device)

        texp = t.view([b, *([1] * len(x1.shape[1:]))])
        # print(texp.shape)
        # z1 = torch.randn_like(x1)
        # zt1 = (1 - texp) * x1 + texp * z1
        # z2 = torch.randn_like(x2)
        # zt2 = (1 - texp) * x2 + texp * z2
        cond_t = torch.cat([t1, t2], dim=0)
        t2 = torch.cat([t, t], dim=0)
        x = torch.cat([x1, x2],dim=0)
        # x = x.clip(0,5)/5 #
        x = x.clip(0,5)
        texp2 = torch.cat([texp, texp], dim=0)
        z = torch.randn_like(x)
        zt = (1 - texp2) * x + texp2 * z
        # z = torch.cat([z1, z2], dim=0) # 噪声
        # zt = torch.cat([zt1, zt2], dim=0) # 中间状态
        cond2 = cond
        if cond2 is not None:
            cond2 = torch.cat([cond, cond], dim=0)
#         print('zt.shape, t.shape', zt.shape, t.shape)
        vtheta = self.model(zt, t2, cond2, cond_t=cond_t)
        batchwise_mse = ((z - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t2, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, x1, t1, t2, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = x1.size(0)
        t_step = 1.0 / sample_steps

        # x1先逆向得到噪声z1，基于条件t1，再正向回来，基于条件t2

        # 第一阶段：从 x1 逆向到噪声 z1
        # 使用条件 t1 进行逆向采样
        # 逆向过程：从 t1 到 1.0（噪声）
        z = x1
        reverse_steps = sample_steps
        for i in range(reverse_steps):
            t_current = i / reverse_steps
            t_current = torch.tensor([t_current] * b, device=z.device)

            # 获取速度场预测
            vc = self.model(z, t_current, cond, cond_t=t1)
            if null_cond is not None:
                vu = self.model(z, t_current, null_cond, cond_t=t1)
                vc = vu + cfg * (vc - vu)

            # 逆向更新：向噪声方向移动
            z = z + t_step * vc


        # 第二阶段：从噪声 z1 正向到目标状态
        # 使用条件 t2 进行正向采样
        images = [z]
        forward_steps = sample_steps
        for i in range(forward_steps):
            t_current = 1 - i / forward_steps
            t_current = torch.tensor([t_current] * b, device=z.device)

            # 获取速度场预测
            vc = self.model(z, t_current, cond, cond_t=t2)
            if null_cond is not None:
                vu = self.model(z, t_current, null_cond, cond_t=t2)
                vc = vu + cfg * (vc - vu)

            # 正向更新：向目标状态移动
            z = z - t_step * vc
            images.append(z)

        return images