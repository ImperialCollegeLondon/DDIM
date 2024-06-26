import math
import torch
from torch import nn
from torch.nn import functional as F


# Sinusoidal position embedding
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Timestep embedding support
class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# Group normalization layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# 3Dtransformer
class TransformerBlock3D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.attn = MultiheadAttention3D(channels, num_heads)
        self.norm2 = nn.GroupNorm(32, channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, d*h*w, c)
        x = x + self.ff(x)
        x = x.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return x
# 3D多头自注意力
class MultiheadAttention3D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, d*h*w, c)

        q = self.query(x).reshape(b, d*h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(b, d*h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(b, d*h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, d*h*w, c)
        out = self.out(out).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        return out

# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Subspace for ULSAM
class SubSpace3D(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.conv_dws = nn.Conv3d(nin, nin, kernel_size=1, stride=1, padding=0, groups=nin)
        self.bn_dws = nn.BatchNorm3d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.conv_point = nn.Conv3d(nin, 1, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn_point = nn.BatchNorm3d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)
        out = self.maxpool(out)
        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)
        m, n, p, q, r = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q, r)
        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = torch.mul(out, x)
        return out + x


# ULSAM 3D
class ULSAM3D(nn.Module):
    def __init__(self, nin, nout, d, h, w, num_splits):
        super().__init__()
        self.nin = nin  # 添加这行
        self.num_splits = num_splits  # 添加这行
        assert nin % num_splits == 0
        self.subspaces = nn.ModuleList([SubSpace3D(int(nin / num_splits)) for _ in range(num_splits)])

    def forward(self, x):
        group_size = int(self.nin / self.num_splits)
        sub_feat = torch.chunk(x, self.num_splits, dim=1)
        out = [self.subspaces[idx](sub_feat[idx]) for idx in range(self.num_splits)]
        return torch.cat(out, dim=1)


# Upsample and Downsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(2, 2, 2), mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


# U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=128, out_channels=1, num_res_blocks=2,
                 attention_resolutions=(8, 16),  transformer_resolutions=(8, 16), dropout=0, channel_mult=(1, 2, 2, 2),
                 conv_resample=True, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.transformer_resolutions = transformer_resolutions if transformer_resolutions else []
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.time_embed = nn.Sequential(nn.Linear(model_channels, model_channels * 4), nn.SiLU(),
                                        nn.Linear(model_channels * 4, model_channels * 4))

        # Initialize down_blocks, middle_block, and up_blocks
        self.down_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1))])
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(model_channels * 2, model_channels * 2, model_channels * 4, dropout),
            ULSAM3D(model_channels * 2, model_channels * 2, 4, 4, 4, 4),
            ResidualBlock(model_channels * 2, model_channels * 2, model_channels * 4, dropout))
        self.up_blocks = nn.ModuleList([])

        # Populate down_blocks and up_blocks
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, model_channels * 4, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(ULSAM3D(ch, ch, ds, ds, ds, 4))
                if ds in transformer_resolutions:
                    layers.append(TransformerBlock3D(ch, num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(ch + down_block_chans.pop(), model_channels * mult, model_channels * 4, dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(ULSAM3D(ch, ch, ds, ds, ds, 4))
                if ds in transformer_resolutions:
                    layers.append(TransformerBlock3D(ch, num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(norm_layer(ch), nn.SiLU(),
                                 nn.Conv3d(model_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, timesteps):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)


# Example usage
unet_model = UNet()
