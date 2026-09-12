"""Multiscale conditional v2 U-Net with local and broad-context attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .model import Block, TimeEmbedding, norm


class AttentionV2(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.q, self.k, self.v, self.proj = [nn.Linear(width, width) for _ in range(4)]

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self.norm(tokens)
        source = q if context is None else self.context_norm(context.flatten(2).transpose(1, 2))
        def split(value):
            return value.reshape(b, -1, self.heads, c//self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(split(self.q(q)), split(self.k(source)), split(self.v(source)), dropout_p=0.)
        attended = attended.transpose(1, 2).reshape(b, h*w, c)
        return ((tokens+self.proj(attended))/math.sqrt(2)).transpose(1, 2).reshape(b, c, h, w)


class UNetV2(nn.Module):
    def __init__(self, condition_channels, base_channels=32, channel_mult=(1, 2, 4, 4),
                 time_dim=128, blocks_per_level=2, attention_heads=4,
                 activation_checkpointing=True, mean_condition=False):
        super().__init__()
        widths = [base_channels*m for m in channel_mult]
        self.checkpointing = activation_checkpointing
        self.mean_condition = mean_condition
        self.time = TimeEmbedding(time_dim)
        nc = condition_channels+(5 if mean_condition else 0)
        self.input = nn.Conv2d(5+nc, widths[0], 3, padding=1)
        self.conditions = nn.ModuleList([nn.Conv2d(nc, w, 1) for w in widths])
        self.down = nn.ModuleList([nn.ModuleList([Block(w, w, time_dim) for _ in range(blocks_per_level)]) for w in widths])
        self.reduce = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths[:-1], widths[1:])])
        self.context = nn.Sequential(nn.Conv2d(condition_channels, base_channels, 3, 2, 1), nn.SiLU(),
                                     nn.Conv2d(base_channels, widths[-1], 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d(8))
        self.self_attention = AttentionV2(widths[-1], attention_heads)
        self.cross_attention = AttentionV2(widths[-1], attention_heads)
        self.up = nn.ModuleList([nn.ModuleList([Block(a+b, b, time_dim)]+
                                    [Block(b, b, time_dim) for _ in range(blocks_per_level-1)])
                                for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.output = nn.Sequential(norm(widths[0]), nn.SiLU(), nn.Conv2d(widths[0], 5, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def run(self, module, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x, time, condition, context, mean=None):
        if self.mean_condition:
            if mean is None:
                raise ValueError('V2 flow requires a frozen regression prediction')
            condition = torch.cat([condition, mean], dim=1)
        temb = self.time(time.float())
        h = self.input(torch.cat([x, condition], dim=1))
        skips = []
        for i, blocks in enumerate(self.down):
            h = (h+self.conditions[i](F.interpolate(condition, size=h.shape[-2:], mode='area')))/math.sqrt(2)
            for block in blocks:
                h = self.run(block, h, temb)
            skips.append(h)
            if i < len(self.reduce):
                h = self.reduce[i](h)
        h = self.run(self.self_attention, h)
        h = self.run(self.cross_attention, h, self.context(context))
        for blocks, skip in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = self.run(block, h, temb)
        return self.output(h)


def regression_v2(model, batch):
    target = batch['target']
    return model(torch.zeros_like(target), torch.zeros(target.shape[0], device=target.device),
                 batch['condition'], batch['context']).float()


@torch.no_grad()
def integrate_v2(model, noise, condition, context, mean, steps):
    if steps < 1:
        raise ValueError('ODE steps must be positive')
    x, dt = noise, 1/steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i*dt, device=x.device)
        k1 = model(x, t, condition, context, mean).float()
        k2 = model(x+dt*k1, t+dt, condition, context, mean).float()
        x = x+dt*(k1+k2)/2
    return x
