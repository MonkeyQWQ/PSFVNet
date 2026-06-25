import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import math


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        return y


class MSTA(nn.Module):
    def __init__(self, channels, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3,
                                    stride=1, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class SIF(nn.Module):
    def __init__(self, channels, expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3,
                               stride=1, padding=1, groups=hidden_channels * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SFB(nn.Module):
    def __init__(self, channels, num_heads=4, expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MSTA(channels, num_heads, bias)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = SIF(channels, expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)


class BPE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        if context is None:
            context = x
        n, c, h, w = context.size()
        x_h = self.pool_h(context)
        x_w = self.pool_w(context).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h


class STAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 5, channels * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            LayerNorm2d(channels)
        )

    def forward(self, frame_feats):
        fused = torch.cat(frame_feats, dim=1)
        return self.fusion(fused)


class AFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1, 1, 0),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_orig = x
        orig_dtype = x.dtype
        if x.dtype == torch.float16: x = x.float()

        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        x_amp_processed = self.freq_conv(x_amp)
        if orig_dtype == torch.float16:
            x_amp_processed = x_amp_processed.float()
            x_phase = x_phase.float()

        x_real = x_amp_processed * torch.cos(x_phase)
        x_imag = x_amp_processed * torch.sin(x_phase)
        x_fft_processed = torch.complex(x_real, x_imag)

        x_freq = torch.fft.irfft2(x_fft_processed, s=(H, W), norm='ortho')
        if orig_dtype == torch.float16: x_freq = x_freq.half()

        gate = self.freq_gate(x_freq)
        return x_orig + x_freq * gate * self.alpha


class RPCE_Module(nn.Module):
    def __init__(self, in_channels=3, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.hyperparam_net = nn.Sequential(
            nn.Conv2d(4, 32, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(32, iterations * 2, 1, 1, 0), nn.Softplus()
        )
        self.atm_net = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, 32, 1, 1, 0), nn.GELU(),
            nn.Conv2d(32, 16, 1, 1, 0), nn.GELU(),
            nn.Conv2d(16, in_channels, 1, 1, 0), nn.Sigmoid()
        )
        self.atm_bpe = BPE(in_channels, in_channels)

        self.trans_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.GELU(),
            nn.Conv2d(16, 1, 1, 1, 0), nn.Sigmoid()
        )

    def forward(self, x, background=None):
        B, C, H, W = x.shape
        atm_init = torch.max(x, dim=2, keepdim=True)[0]
        atm_init = torch.max(atm_init, dim=3, keepdim=True)[0]
        atm = atm_init.expand(B, C, H, W)

        mask = torch.where(x > 0.6, torch.zeros((B, 1, H, W), device=x.device),
                          torch.ones((B, 1, H, W), device=x.device))
        mask = (mask[:, 0:1, :, :] + mask[:, 1:2, :, :] + mask[:, 2:3, :, :]) / 3

        lambda_params = self.hyperparam_net(torch.cat([F.adaptive_avg_pool2d(mask, 1),
                                                      F.adaptive_avg_pool2d(atm_init, 1)], dim=1)) + 1e-6
        atm_list, trans_list = [], []

        for i in range(self.iterations):
            atm_prior = self.atm_net(x)
            atm_enhanced = self.atm_bpe(atm_prior.expand(B, C, H, W), background) if background is not None else atm_prior.expand(B, C, H, W)
            trans = self.trans_net(x)

            atm = atm_enhanced * 0.8 + atm * 0.2
            trans = trans * 0.9 + mask * 0.1
            atm_list.append(atm)
            trans_list.append(trans)

        return atm_list, trans_list


class GCM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1, 1, 0),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.PixelShuffle(2)
        )
    def forward(self, x): return self.body(x)


class CoarseRecoveryNet(nn.Module):
    def __init__(self, in_channels=3, num_frames=5, base_channels=48,
                 num_blocks=[2, 2, 4], num_heads=[1, 2, 4]):
        super().__init__()
        self.physical_prior = RPCE_Module(in_channels, iterations=2)
        self.feat_extract = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.context = GCM(base_channels * 4)

        self.encoder_l1 = nn.Sequential(*[SFB(base_channels, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = Downsample(base_channels)
        self.encoder_l2 = nn.Sequential(
            *[SFB(base_channels * 2, num_heads[1]) for _ in range(num_blocks[1])],
            AFM(base_channels * 2)
        )
        self.down2 = Downsample(base_channels * 2)
        self.encoder_l3 = nn.Sequential(
            *[SFB(base_channels * 4, num_heads[2]) for _ in range(num_blocks[2])],
            AFM(base_channels * 4)
        )

        self.temporal_fusion = STAM(base_channels * 4)
        self.up2 = Upsample(base_channels * 4)
        self.skip_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, 1, 0)
        self.decoder_l2 = nn.Sequential(*[SFB(base_channels * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.up1 = Upsample(base_channels * 2)
        self.skip_conv1 = nn.Conv2d(base_channels, base_channels, 1, 1, 0)
        self.decoder_l1 = nn.Sequential(*[SFB(base_channels, num_heads[0]) for _ in range(num_blocks[0])])
        self.output = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        mid_idx = T // 2
        mid_frame = x[:, :, mid_idx, :, :]

        atm_list, trans_list = self.physical_prior(mid_frame)
        context_feat = self.context(F.interpolate(mid_frame, scale_factor=0.25))

        frame_feats_l1, frame_feats_l2, frame_feats_l3 = [], [], []
        for t in range(T):
            feat = self.feat_extract(x[:, :, t, :, :])
            enc1 = self.encoder_l1(feat); frame_feats_l1.append(enc1)
            enc2 = self.encoder_l2(self.down1(enc1)); frame_feats_l2.append(enc2)
            enc3 = self.encoder_l3(self.down2(enc2) + context_feat); frame_feats_l3.append(enc3)

        fused_feat = self.temporal_fusion(frame_feats_l3)

        dec2 = self.decoder_l2(self.up2(fused_feat) + self.skip_conv2(frame_feats_l2[mid_idx]))
        dec1 = self.decoder_l1(self.up1(dec2) + self.skip_conv1(frame_feats_l1[mid_idx]))

        network_out = self.output(dec1)
        atm, trans = atm_list[-1], trans_list[-1]
        physical_out = torch.clamp((mid_frame - atm * (1 - trans)) / (trans + 1e-6), 0, 1)

        return torch.clamp(0.6 * network_out + 0.4 * physical_out, 0, 1), atm_list, trans_list


class FineRefinementNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_blocks=4):
        super().__init__()
        self.feat_extract = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.encoder = nn.Sequential(
            *[SFB(base_channels, num_heads=2) for _ in range(num_blocks)],
            AFM(base_channels)
        )
        self.decoder = nn.Sequential(*[SFB(base_channels, num_heads=2) for _ in range(num_blocks)])
        self.output = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.feat_extract(x)
        feat = self.decoder(self.encoder(feat))
        return torch.clamp(x + self.output(feat), 0, 1)


class SFVNet(nn.Module):
    def __init__(self, in_channels=3, num_frames=5, base_channels=48,
                 num_blocks=[2, 2, 4], num_heads=[1, 2, 4], use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.coarse_net = CoarseRecoveryNet(in_channels, num_frames, base_channels, num_blocks, num_heads)
        self.fine_net = FineRefinementNet(in_channels, 32, 4)

    def forward(self, x, return_intermediate=False):
        if self.use_checkpoint and self.training:
            coarse_out, atm_list, trans_list = checkpoint.checkpoint(self.coarse_net, x, use_reentrant=False)
            fine_out = checkpoint.checkpoint(self.fine_net, coarse_out, use_reentrant=False)
        else:
            coarse_out, atm_list, trans_list = self.coarse_net(x)
            fine_out = self.fine_net(coarse_out)

        if return_intermediate:
            return fine_out, coarse_out, atm_list[-1], trans_list[-1]
        return fine_out


def build_sfv_net(num_frames=5, base_channels=48, num_blocks=[2, 2, 4],
                  num_heads=[1, 2, 4], use_checkpoint=False):
    return SFVNet(in_channels=3, num_frames=num_frames, base_channels=base_channels,
                  num_blocks=num_blocks, num_heads=num_heads, use_checkpoint=use_checkpoint)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_sfv_net(num_frames=5, base_channels=48, num_blocks=[2, 2, 4],
                          num_heads=[1, 2, 4], use_checkpoint=False).to(device)

    model.eval()
    x = torch.randn(2, 3, 5, 224, 224).to(device)
    with torch.no_grad():
        output = model(x, return_intermediate=True)