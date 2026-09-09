"""Conditional residual U-Net for a straight-path flow-matching velocity."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width*4), nn.SiLU(), nn.Linear(width*4, width))

    def forward(self, time):
        freq = torch.exp(-math.log(10000)*torch.arange(self.width//2, device=time.device)/(self.width//2-1))
        phase = time[:, None]*freq[None]*1000
        return self.mlp(torch.cat([phase.sin(), phase.cos()], dim=-1))


class Block(nn.Module):
    def __init__(self, cin, cout, time_dim):
        super().__init__()
        self.n1, self.n2 = norm(cin), norm(cout)
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.time = nn.Linear(time_dim, 2*cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.time(F.silu(t)).chunk(2, dim=1)
        h = self.n2(h)*(1+scale[:, :, None, None])+shift[:, :, None, None]
        return (self.c2(F.silu(h))+self.skip(x))/math.sqrt(2)


class VelocityUNet(nn.Module):
    def __init__(self, condition_channels, base_channels=48, channel_mult=(1, 2, 4, 4), time_dim=128, activation_checkpointing=True):
        super().__init__()
        if time_dim < 4 or time_dim % 2:
            raise ValueError('time_dim must be even and >=4')
        widths = [base_channels*m for m in channel_mult]
        self.checkpointing = activation_checkpointing
        self.time = TimeEmbedding(time_dim)
        self.input = nn.Conv2d(4+condition_channels, widths[0], 3, padding=1)
        self.down = nn.ModuleList([Block(w, w, time_dim) for w in widths])
        self.reduce = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths[:-1], widths[1:])])
        self.mid = Block(widths[-1], widths[-1], time_dim)
        self.up = nn.ModuleList([Block(a+b, b, time_dim) for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.output = nn.Sequential(norm(widths[0]), nn.SiLU(), nn.Conv2d(widths[0], 4, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def block(self, block, x, time):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, time, use_reentrant=False)
        return block(x, time)

    def forward(self, x, time, condition):
        temb = self.time(time.float())
        h = self.input(torch.cat([x, condition], dim=1))
        skips = []
        for i, block in enumerate(self.down):
            h = self.block(block, h, temb)
            skips.append(h)
            if i < len(self.reduce):
                h = self.reduce[i](h)
        h = self.block(self.mid, h, temb)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = self.block(block, torch.cat([h, skip], dim=1), temb)
        return self.output(h)


def flow_loss(model, batch, halo, channel_weights, generator=None):
    x1, cond = batch['target'], batch['condition']
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    time = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    xt = (1-time[:, None, None, None])*x0 + time[:, None, None, None]*x1
    error = (model(xt, time, cond).float()-(x1-x0)).square()
    if halo:
        error = error[:, :, halo:-halo, halo:-halo]
    weights = torch.as_tensor(channel_weights, device=error.device, dtype=error.dtype)[None, :, None, None]
    area = batch['area'][:, None]
    return (error*weights*area).sum()/(area.sum()*weights.sum())


@torch.no_grad()
def integrate(model, noise, condition, steps):
    """Fixed-step Heun integration of dx/dt=v; random initial noise gives members."""
    if steps < 1:
        raise ValueError('ODE steps must be positive')
    x = noise
    dt = 1/steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i*dt, device=x.device)
        k1 = model(x, t, condition).float()
        k2 = model(x+dt*k1, t+dt, condition).float()
        x = x+dt*(k1+k2)/2
    return x
