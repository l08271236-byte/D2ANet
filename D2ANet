# models/D2ANet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from thop import profile
from .smt import smt_b  

# -------------------------
# Basic blocks
# -------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, relu=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding if padding is not None else dilation,
                              dilation=dilation, bias=not bn)
        self.bn = nn.BatchNorm2d(out_planes) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if relu else nn.Identity()
    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, d=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, padding=d if k==3 else 0, dilation=d, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.op(x)

# -------------------------
# Sobel for HBE
# -------------------------
def _make_sobel_kernel(in_chan: int, out_chan: int, kernel_3x3: np.ndarray) -> torch.Tensor:
    k = kernel_3x3.reshape((1, 1, 3, 3)).astype(np.float32)
    k = np.repeat(k, in_chan, axis=1)
    k = np.repeat(k, out_chan, axis=0)
    return torch.from_numpy(k)

def fixed_sobel(in_chan: int, out_chan: int) -> Tuple[nn.Conv2d, nn.Conv2d]:
    kx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]], dtype=np.float32)
    conv_x = nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False)
    conv_y = nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False)
    conv_x.weight.data = _make_sobel_kernel(in_chan, out_chan, kx)
    conv_y.weight.data = _make_sobel_kernel(in_chan, out_chan, ky)
    conv_x.weight.requires_grad = False
    conv_y.weight.requires_grad = False
    return conv_x, conv_y

def sobel_gate(conv_x: nn.Conv2d, conv_y: nn.Conv2d, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gx = conv_x(x)
    gy = conv_y(x)
    g = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
    return torch.sigmoid(g) * x  # gate

# -------------------------
# Channel / Spatial / Frequency attentions for DDA
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pre = CBR(ch, ch, k=3)
        mid = max(1, ch // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False)
        )
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
    def forward(self, x):
        xc = self.pre(x)
        avg = self.mlp(F.adaptive_avg_pool2d(xc, 1))
        max = self.mlp(F.adaptive_max_pool2d(xc, 1))
        w = torch.sigmoid(self.proj(avg + max))
        return xc * w

class SpatialAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pre = CBR(ch, ch, k=3)
        mid = max(1, ch // reduction)
        self.to_mid = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True))
        self.mix = nn.Conv2d(2*mid, ch, 1, bias=False)
    def forward(self, x):
        xs = self.pre(x)
        avg = self.to_mid(F.adaptive_avg_pool2d(xs, 1))
        max = self.to_mid(F.adaptive_max_pool2d(xs, 1))
        w = torch.sigmoid(self.mix(torch.cat([avg, max], dim=1)))
        return xs * w

class AttentionModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention(ch)
        self.fuse = nn.Conv2d(2*ch, ch, 1, bias=False)
    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(x)
        fused = self.fuse(torch.cat([ca, sa], dim=1))
        return x + fused

class FrequencyAttention(nn.Module):
    def __init__(self, ch, pool_size=(8,8), hidden_ratio=8, gamma_init=0.1):
        super().__init__()
        hid = max(1, ch // hidden_ratio)
        self.proc = nn.Sequential(
            nn.Conv2d(ch, hid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.pool_size = pool_size
        self.gamma = nn.Parameter(torch.tensor(gamma_init), requires_grad=True)
    def forward(self, x):
        B, C, H, W = x.shape
        mag = torch.abs(torch.fft.fft2(x, norm='ortho'))
        mag_s = F.adaptive_avg_pool2d(mag, self.pool_size)
        att_s = self.proc(mag_s)
        att = F.interpolate(att_s, size=(H, W), mode='bilinear', align_corners=False)
        return x * (1.0 + self.gamma * att)

# -------------------------
# Hybrid Boundary Extractor
# -------------------------
class HBE(nn.Module):
    """
      - High-level guidance: Rh = Up( Conv3x3( Cat(R3, Up(R4)) ) )  -> up to R1 size
      - Shallow gating (Sobel on R2/R1), align & modulate with Rh, fuse to R_el
      - Boundary prior: E = sigmoid( Conv3x3(|R_el - Rh|)  )
    """
    def __init__(self, ch=64):
        super().__init__()
        # High-level guidance
        self.conv_high = BasicConv2d(2*ch, ch, 3, padding=1)

        # Sobel（R2, R1）
        self.sobel2_x, self.sobel2_y = fixed_sobel(ch, 1)
        self.sobel1_x, self.sobel1_y = fixed_sobel(ch, 1)

        # shallow with Rh and fuse
        self.conv_mod = BasicConv2d(ch, ch, 3, padding=1)    
        self.conv_fuse_low = BasicConv2d(2*ch, ch, 3, padding=1)

        # edge logits
        self.conv_edge = BasicConv2d(ch, 1, 3, padding=1)  

    def forward(self, r1, r2, r3, r4):
        # Rh
        r4u_r3 = F.interpolate(r4, size=r3.shape[-2:], mode='bilinear', align_corners=False)
        r34 = self.conv_high(torch.cat([r3, r4u_r3], dim=1))
        rh = F.interpolate(r34, size=r1.shape[-2:], mode='bilinear', align_corners=False)

        # R_el
        g2 = sobel_gate(self.sobel2_x, self.sobel2_y, r2)  
        g1 = sobel_gate(self.sobel1_x, self.sobel1_y, r1)

        g2u = F.interpolate(g2, size=rh.shape[-2:], mode='bilinear', align_corners=False)
        g1u = F.interpolate(g1, size=rh.shape[-2:], mode='bilinear', align_corners=False)

        s2 = self.conv_mod(g2u * rh + g2u)
        s1 = self.conv_mod(g1u * rh + g1u)

        r_el = self.conv_fuse_low(torch.cat([s2, s1], dim=1))

        # edge prior
        E = torch.sigmoid (self.conv_edge(torch.abs(r_el - rh)))

        return E, rh, r_el

# -------------------------
# Edge-Guided Multiscale Interaction Module
# -------------------------
class EGMI(nn.Module):

    def __init__(self, ch=64, dilation=5):
        super().__init__()
        self.d = dilation
        self.adj = nn.ModuleList([BasicConv2d(ch, ch, 3, padding=1) for _ in range(3)])
        self.a1 = nn.ModuleList([BasicConv2d(ch, ch, 3, padding=1) for _ in range(3)])
        self.a2 = nn.ModuleList([BasicConv2d(ch, ch, 3, padding=1) for _ in range(3)])
        self.b1 = nn.ModuleList([BasicConv2d(ch, ch, (1,3), padding=(0,1)) for _ in range(3)])
        self.b2 = nn.ModuleList([BasicConv2d(ch, ch, (3,1), padding=(1,0)) for _ in range(3)])
        self.b3 = nn.ModuleList([BasicConv2d(ch, ch, 3, padding=dilation, dilation=dilation) for _ in range(3)])
        self.fuse = nn.ModuleList([BasicConv2d(2*ch, ch, 3, padding=1) for _ in range(3)])
        self.out = nn.ModuleList([BasicConv2d(ch, ch, 3, padding=1) for _ in range(3)])

    def _pair(self, r_low, r_high):
        rhu = F.interpolate(r_high, size=r_low.shape[-2:], mode='bilinear', align_corners=False)
        return r_low + rhu

    def forward(self, r1, r2, r3, r4, E):
        R = [r1, r2, r3, r4]
        Ys = []
        for i in range(3):  
            Mi_in = self._pair(R[i], R[i+1])
            Mi = self.adj[i](Mi_in)

            Ai = self.a2[i](self.a1[i](Mi))
            Bi = self.b3[i](self.b2[i](self.b1[i](Mi)))
            Ti = self.fuse[i](torch.cat([Ai, Bi], dim=1))

            Zi = Ti + Mi
            Eu = F.interpolate(E, size=Zi.shape[-2:], mode='bilinear', align_corners=False)
            Yi = self.out[i](Zi * Eu + Zi)
            Ys.append(Yi)
        return Ys[0], Ys[1], Ys[2]

# -------------------------
# Dual-domain Attention Module
# -------------------------
class DDA(nn.Module):
    """
    Dual-domain Attention:
       Z1 = Conv3x3(Cat(Y2, Up(Y3))) -> AM(Z1), FA(Z1) -> Cat -> 1x1 -> residual -> O1
       Z2 = Conv3x3(Cat(Y1, Up(o1))) -> AM(Z2), FA(Z2) -> Cat -> 1x1 -> residual -> O2
    """
    def __init__(self, ch=64, fa_pool=(8,8), hidden_ratio=8):
        super().__init__()

        self.fuse1 = BasicConv2d(2*ch, ch, 3, padding=1)
        self.am1 = AttentionModule(ch)
        self.fa1 = FrequencyAttention(ch, pool_size=fa_pool, hidden_ratio=hidden_ratio)
        self.mix1 = nn.Conv2d(2*ch, ch, 1, bias=False)

        self.fuse2 = BasicConv2d(2*ch, ch, 3, padding=1)
        self.am2 = AttentionModule(ch)
        self.fa2 = FrequencyAttention(ch, pool_size=fa_pool, hidden_ratio=hidden_ratio)
        self.mix2 = nn.Conv2d(2*ch, ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, y1, y2, y3):

        z1 = self.fuse1(torch.cat([y2, F.interpolate(y3, size=y2.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        t1 = self.mix1(torch.cat([self.am1(z1), self.fa1(z1)], dim=1))
        o1 = self.act(z1 + t1)

        z2 = self.fuse2(torch.cat([y1, F.interpolate(o1, size=y1.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        t2 = self.mix2(torch.cat([self.am2(z2), self.fa2(z2)], dim=1))
        o2 = self.act(z2 + t2)
        return o1, o2 

# -------------------------
# D2ANet (main)
# -------------------------
class D2ANet(nn.Module):
    """
    D2ANet = HBE (edge prior) + EGMI (adjacent interaction) + DDA (dual-domain attention)
    Backbone: smt_b -> (r1,r2,r3,r4), then 1x1 conv to make all channels=64
    Heads: three logits (final + two aux), + edge map
    """
    def __init__(self):
        super().__init__()

        self.backbone = smt_b()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2) 

        # project r2,r3,r4 to 64 channels
        self.proj2 = BasicConv2d(128, 64, 1, padding=0)
        self.proj3 = BasicConv2d(256, 64, 1, padding=0)
        self.proj4 = BasicConv2d(512, 64, 1, padding=0)

        # three modules
        self.hbe = HBE(ch=64)
        self.egmi = EGMI(ch=64, dilation=5)
        self.dda = DDA(ch=64, fa_pool=(8,8), hidden_ratio=8)

        # prediction heads
        self.pred_final = nn.Conv2d(64, 1, 3, padding=1)
        self.pred_s1 = nn.Conv2d(64, 1, 3, padding=1)
        self.pred_y3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x: torch.Tensor):
        # backbone
        r1, r2, r3, r4 = self.backbone(x)  
        r2 = self.proj2(r2)
        r3 = self.proj3(r3)
        r4 = self.proj4(r4)
        # HBE: edge prior
        E, Rh, Rel = self.hbe(r1, r2, r3, r4)
        # EGMI: adjacent interaction + edge-guided refinement -> Y1,Y2,Y3
        Y1, Y2, Y3 = self.egmi(r1, r2, r3, r4, E)
        # DDA: two output   (Y3,Y2)->O1,  (O1,Y1)->O2
        O1, O2 = self.dda(Y1, Y2, Y3)

        # outputs
        final_size = x.shape[-2:]
        out1 = F.interpolate(self.pred_final(O2), size=final_size, mode='bilinear', align_corners=False) 
        out2 = F.interpolate(self.pred_s1(O1), size=final_size, mode='bilinear', align_corners=False)  
        out3 = F.interpolate(self.pred_y3(Y3), size=final_size, mode='bilinear', align_corners=False)  
        edge = F.interpolate(E, size=final_size, mode='bilinear', align_corners=False)     

        return out1, out2, out3, edge

    def load_pre(self, pre_model):
        checkpoint = torch.load(pre_model, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            self.backbone.load_state_dict(checkpoint['model'])
        else:
            try:
                self.backbone.load_state_dict(checkpoint)
            except Exception:
                print("Warning: failed to load checkpoint into backbone; check checkpoint format.")
        print(f"loading pre_model {pre_model}")

# -------------------------
# quick test
# -------------------------
if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    model = D2ANet().eval()
    with torch.no_grad():
        o1, o2, o3, e = model(x)
    print("out shapes:", o1.shape, o2.shape, o3.shape, e.shape)
    try:
        flops, params = profile(model, inputs=(x,), verbose=False)
        print('FLOPs: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))
    except Exception as ex:
        print("thop profile skipped:", ex)

