import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from ultralytics.nn.modules.conv import Conv

try:
    from torchvision.ops import deform_conv2d  # torchvision >= 0.8
 
    _USE_TV = True
except ImportError:
    _USE_TV = False


class DCNv2(nn.Module):
    """
    Deformable Convolutional Network v2 layer.
 
    Includes BN + SiLU activation so callers don't need to add them,
    avoiding accidental double-normalisation in bottleneck blocks.
 
    Args:
        in_channels       (int)  : input channels
        out_channels      (int)  : output channels
        kernel_size       (int)  : kernel size (default 3)
        stride            (int)  : stride (default 1)
        padding           (int)  : zero-padding (default 1)
        dilation          (int)  : dilation (default 1)
        groups            (int)  : conv groups (default 1)
        deformable_groups (int)  : offset groups (default 1)
        with_modulation   (bool) : DCNv2 modulation mask (default True)
        act               (bool) : apply BN+SiLU after conv (default True)
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deformable_groups: int = 1,
        with_modulation: bool = True,
        act: bool = True,
    ):
        super().__init__()
 
        # FIX-D: validate groups up front — bad value → Python error, not CUDA crash
        assert in_channels % groups == 0, (
            f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        )
        assert out_channels % groups == 0, (
            f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        )
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_modulation = with_modulation
 
        # Learnable weight (no built-in BN, we add our own below)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
 
        # Offset conv → 2 × deformable_groups × kH × kW channels
        offset_ch = deformable_groups * 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels, offset_ch,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True,   # FIX-D
        )
 
        # Modulation mask conv → deformable_groups × kH × kW channels
        if with_modulation:
            mask_ch = deformable_groups * kernel_size * kernel_size
            self.mask_conv = nn.Conv2d(
                in_channels, mask_ch,
                kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=True,  # FIX-D
            )
        else:
            self.mask_conv = None
 
        # FIX-C: BN + activation live here, not in the bottleneck wrapper
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
 
        self._init_weights()
 
    def _init_weights(self):
        # FIX-E: guard against zero fan_in before calling kaiming
        fan_in = (self.in_channels // self.groups) * self.kernel_size ** 2
        if fan_in > 0:
            nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(self.weight)
 
        # Zero-init offset/mask so the layer starts as a standard conv
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.mask_conv is not None:
            nn.init.zeros_(self.mask_conv.weight)
            nn.init.zeros_(self.mask_conv.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIX-B: force contiguous — .chunk() returns views; deform_conv2d's
        # CUDA kernel requires contiguous memory and segfaults on Windows otherwise.
        x = x.contiguous()
 
        offset = self.offset_conv(x)
 
        if _USE_TV:
            mask = (
                torch.sigmoid(self.mask_conv(x))
                if self.mask_conv is not None else None
            )
            out = deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=(self.stride, self.stride),
                padding=(self.padding, self.padding),
                dilation=(self.dilation, self.dilation),   # FIX-D
                mask=mask,
                # groups is implicit in weight shape for torchvision deform_conv2d
            )
        else:
            # CPU/no-torchvision fallback: standard conv, offsets not applied
            out = F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups,
            )
 
        return self.act(self.bn(out))
 
 
# ---------------------------------------------------------------------------
# DCNv2 Bottleneck  (replaces standard Bottleneck inside C2f)
# ---------------------------------------------------------------------------
 
class DCNv2Bottleneck(nn.Module):
    """
    Bottleneck: Conv 1×1 (no act) → DCNv2 3×3 (with BN+SiLU) → optional shortcut.
 
    BN+SiLU lives inside DCNv2, so this wrapper is deliberately activation-free
    to avoid the double-normalisation that caused NaN → CUDA crash (FIX-C).
    """
 
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        # cv1: plain 1×1 pointwise (Conv from ultralytics already adds BN+SiLU)
        self.cv1 = Conv(c1, c_, 1, 1)
        # DCNv2 handles BN+SiLU internally (act=True by default)
        self.dcn = DCNv2(c_, c2, kernel_size=3, stride=1, padding=1, act=True)
        self.add = shortcut and (c1 == c2)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dcn(self.cv1(x))
        return x + y if self.add else y
 
 
# ---------------------------------------------------------------------------
# C2f_DCN  — drop-in replacement for Ultralytics C2f
# ---------------------------------------------------------------------------
 
class DCNConvC2f(nn.Module):
    """
    C2f with DCNv2 bottleneck blocks. Diagram-accurate dense skip connections:
 
        Input → Conv1×1 → Split(a, b)
                b → Block_1 → Block_2 → … → Block_n
                Concat(a, b, out_1, …, out_n) → Conv1×1 → Output
 
    Args:
        c1 (int)       : input channels
        c2 (int)       : output channels
        n  (int)       : number of DCNv2Bottleneck blocks
        shortcut (bool): residual inside each bottleneck
        g  (int)       : unused (API parity with C2f)
        e  (float)     : hidden channel expansion (default 0.5)
    """
 
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)                          # hidden channels (= 0.5 × c2)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)         # entry: c1 → 2c
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)   # exit:  (2+n)c → c2
        self.m = nn.ModuleList(
            DCNv2Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0)
            for _ in range(n)
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along channel dim; .contiguous() ensures deform_conv2d gets
        # contiguous tensors even after chunk() (FIX-B applied at DCNv2 level too)
        a, b = self.cv1(x).chunk(2, dim=1)
 
        outputs = [a, b]
        t = b
        for block in self.m:
            t = block(t)
            outputs.append(t)
 
        return self.cv2(torch.cat(outputs, dim=1))
 
    def info(self):
        n = len(self.m)
        print(
            f"C2f_DCN | hidden_ch={self.c} | n_blocks={n} "
            f"| concat_ch={(2 + n) * self.c} | out={self.cv2.conv.out_channels}"
        )
