import math
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from ultralytics.nn.modules.conv import Conv


class DeformConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self, c1, c2, k=3, s=1, p: int = None, g=1, d=1, act=True, shape_init=True
    ):
        super().__init__()
        self.k = k
        self.s = s
        self.d = d
        self.g = g
        self.p = k // 2 if p is None else p

        # base weight
        self.conv = nn.Conv2d(c1, c2, k, s, self.p, groups=g, dilation=d, bias=False)

        # offset [c1, 2*k*k] + mask [c1, k*k]
        self.offset_conv = nn.Conv2d(
            c1, 3 * k * k, k, s, self.p, groups=g, dilation=d, bias=True
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

        self._init_weights(shape_init)

    def _init_weights(self, shape_init: bool):
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

        if shape_init:
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
        else:
            nn.init.kaiming_uniform_(self.offset_conv.weight, a=math.sqrt(5))
            nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        off_mask = self.offset_conv(x)

        N = 2 * self.k * self.k
        offset = off_mask[:, :N].contiguous()  # (B, 2k^2, H, W)
        mask = torch.sigmoid(off_mask[:, N:]).contiguous()  # (B, k^2, H, W)

        print(f"x: {x.shape}, offset: {offset.shape}, mask: {mask.shape}, w: {self.conv.weight.shape}")

        out = deform_conv2d(
            x,
            offset,
            self.conv.weight,
            mask=mask,
            bias=None,
            stride=self.s,
            padding=self.p,
            dilation=self.d,
        )

        return self.act(out)


class DCNBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, shape_init=True):
        super().__init__()

        c_ = int(c2 * e)
        self.cv1 = DeformConv(c1, c_, k[0], shape_init=shape_init)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv1(self.cv2(x))
        return x + out if self.add else out


class DCNConvC2f(nn.Module):
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=3, shape_init=True
    ):
        super().__init__()

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            DCNBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=(kernel_size, kernel_size),
                e=1.0,
                shape_init=shape_init,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
