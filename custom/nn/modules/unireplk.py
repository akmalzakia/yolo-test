"""
c3_unireplkblock.py
===================
C3_UniRepLKBlock — A C3-structured CSP module that replaces the vanilla
Bottleneck stack with UniRepLKNetBlock (CVPR 2024 / TPAMI 2025).

Architecture (mirrors the diagram from RLK-net, adapted for UniRepLKNet):

                ┌─ CBS(S=1,K=1) ─────────────────────────────┐
    input ──────┤                                              ├── Concat ── CBS(S=1,K=1) ── output
                └─ CBS(S=1,K=1) ── UniRepLKNetBlock × N ──────┘

vs vanilla C3:
                ┌─ CBS(S=1,K=1) ─────────────────────────────┐
    input ──────┤                                              ├── Concat ── CBS(S=1,K=1) ── output
                └─ CBS(S=1,K=1) ── Bottleneck × N ────────────┘

Key design choices
------------------
UniRepLKNetBlock internal structure (per original paper):
    x ──► DW-LargeKernel (DilatedReparamBlock) ──► BN ──► SE
      ──► Linear(expand) ──► GELU ──► GRN ──► Linear(project) ──► BN
      ──► LayerScale ──► DropPath ──► + residual

The block is designed as a *channel-preserving* operator (dim in == dim out),
which maps perfectly onto C3's hidden channel (`c_`) on both branches.

UniRepLKNetBlock supports:
  - `deploy=True`:  fuses BN+DilatedReparam for fast inference (reparameterize())
  - `kernel_size`:  3 / 5 (small) or 7 / 9 / 11 / 13 / 15 / 17 (large-kernel)
  - `with_cp`:      gradient checkpointing to save GPU memory
  - `drop_path`:    stochastic depth regularisation

Usage in YAML (after registering in tasks.py)
---------------------------------------------
  - [-1, 3, C3_UniRepLKBlock, [256, True]]            # defaults: kernel=13
  - [-1, 6, C3_UniRepLKBlock, [512, False, 1, 0.5, 13]]  # explicit kernel_size
  - [-1, 3, C3_UniRepLKBlock, [256, True, 1, 0.5, 7]]    # smaller kernel, lighter

tasks.py registration (lazy import, same pattern as other custom modules):
    from c3_unireplkblock import C3_UniRepLKBlock, CUSTOM_UNIREPLK_MODULES, CUSTOM_UNIREPLK_REPEAT

Source: https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py  (Apache 2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath          # older timm

from ultralytics.nn.modules.conv import Conv          # CBS: Conv-BN-SiLU


# ============================================================================
# Helpers extracted from unireplknet.py (self-contained, no full-file import)
# ============================================================================

def _get_conv2d(in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias):
    """
    Standard nn.Conv2d wrapper.  The iGEMM large-kernel optimisation from the
    original repo is optional and requires a custom CUDA extension; we omit it
    here for portability — swap in DepthWiseConv2dImplicitGEMM if installed.
    """
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=bias
    )


def _get_bn(dim, use_sync_bn=False):
    return nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)


def _fuse_bn(conv, bn):
    """Fuse a Conv2d + BN into a single Conv2d (used during reparameterisation)."""
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (
        conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
        bn.bias + (conv_bias - bn.running_mean) * bn.weight / std,
    )


def _convert_dilated_to_nondilated(kernel, dilate_rate):
    """Expand a dilated kernel to its equivalent dense kernel."""
    identity = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)
    if kernel.size(1) == 1:                           # depth-wise
        return F.conv_transpose2d(kernel, identity, stride=dilate_rate)
    slices = [
        F.conv_transpose2d(kernel[:, i:i+1], identity, stride=dilate_rate)
        for i in range(kernel.size(1))
    ]
    return torch.cat(slices, dim=1)


def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equiv_k = dilated_r * (dilated_k - 1) + 1
    equiv_kernel = _convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    pad = large_k // 2 - equiv_k // 2
    return large_kernel + F.pad(equiv_kernel, [pad] * 4)


# ============================================================================
# GRN, NCHWtoNHWC, NHWCtoNCHW  (unchanged from official repo)
# ============================================================================

class GRNwithNHWC(nn.Module):
    """Global Response Normalisation (ConvNeXt V2) — input layout (N,H,W,C)."""

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim)) if use_bias else None

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = (self.gamma * Nx + 1) * x
        return out + self.beta if self.use_bias else out


class NCHWtoNHWC(nn.Module):
    def forward(self, x): return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x): return x.permute(0, 3, 1, 2)


# ============================================================================
# SEBlock  (unchanged from official repo)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation for (N,C,H,W) tensors."""

    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, bias=True)
        self.up   = nn.Conv2d(internal_neurons, input_channels, 1, bias=True)
        self.c    = input_channels

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.up(F.relu(self.down(s), inplace=True)))
        return x * s.view(-1, self.c, 1, 1)


# ============================================================================
# DilatedReparamBlock  (unchanged from official repo, iGEMM path removed)
# ============================================================================

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparameterisation Block — the large-kernel depthwise conv at the
    heart of UniRepLKNet.

    In training mode:  large-kernel conv + several dilated small-kernel branches.
    At deploy time:    all branches are fused into a single large-kernel conv
                       via reparameterisation (call merge_dilated_branches()).

    Supported kernel sizes and their default branch configs:
        17 → [5×1, 9×2, 3×4, 3×5, 3×7]
        15 → [5×1, 7×2, 3×3, 3×5, 3×7]
        13 → [5×1, 7×2, 3×3, 3×4, 3×5]   ← default used in YOLO context
        11 → [5×1, 5×2, 3×3, 3×4, 3×5]
         9 → [5×1, 5×2, 3×3, 3×4]
         7 → [5×1, 3×2, 3×3]
         5 → [3×1, 3×2]
    """

    _CONFIGS = {
        17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
        15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
         9: ([5, 5, 3, 3],    [1, 2, 3, 4]),
         7: ([5, 3, 3],       [1, 2, 3]),
         5: ([3, 3],          [1, 2]),
    }

    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False):
        super().__init__()
        if kernel_size not in self._CONFIGS:
            raise ValueError(
                f"DilatedReparamBlock only supports kernel sizes "
                f"{list(self._CONFIGS.keys())}, got {kernel_size}."
            )
        self.lk_origin = _get_conv2d(
            channels, channels, kernel_size,
            stride=1, padding=kernel_size // 2,
            dilation=1, groups=channels, bias=deploy,
        )
        self.kernel_sizes, self.dilates = self._CONFIGS[kernel_size]

        if not deploy:
            self.origin_bn = _get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                pad = (r * (k - 1) + 1) // 2
                self.__setattr__(
                    f"dil_conv_k{k}_{r}",
                    nn.Conv2d(channels, channels, k, stride=1,
                              padding=pad, dilation=r, groups=channels, bias=False),
                )
                self.__setattr__(
                    f"dil_bn_k{k}_{r}",
                    _get_bn(channels, use_sync_bn),
                )

    def forward(self, x):
        if not hasattr(self, "origin_bn"):       # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f"dil_conv_k{k}_{r}")
            bn   = self.__getattr__(f"dil_bn_k{k}_{r}")
            out  = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        """Fuse all branches → single large-kernel conv (call before deploy)."""
        if not hasattr(self, "origin_bn"):
            return                                 # already merged
        origin_k, origin_b = _fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f"dil_conv_k{k}_{r}")
            bn   = self.__getattr__(f"dil_bn_k{k}_{r}")
            bk, bb = _fuse_bn(conv, bn)
            origin_k = _merge_dilated_into_large_kernel(origin_k, bk, r)
            origin_b = origin_b + bb

        C = origin_k.size(0)
        K = origin_k.size(2)
        merged = _get_conv2d(C, C, K, stride=1, padding=K // 2,
                             dilation=1, groups=C, bias=True)
        merged.weight.data = origin_k
        merged.bias.data   = origin_b
        self.lk_origin = merged
        del self.origin_bn
        for k, r in zip(self.kernel_sizes, self.dilates):
            del self.__dict__["_modules"][f"dil_conv_k{k}_{r}"]
            del self.__dict__["_modules"][f"dil_bn_k{k}_{r}"]


# ============================================================================
# UniRepLKNetBlock  (faithful re-implementation — portable, no timm registry)
# ============================================================================

class UniRepLKNetBlock(nn.Module):
    """
    Single UniRepLKNetBlock.

    Internal data flow (training mode):
        x ──► DilatedReparamBlock (large depthwise conv) ──► BN ──► SEBlock
          ──► pwconv1 (Linear expand, NHWC) ──► GELU ──► GRN
          ──► pwconv2 (Linear project, NHWC → NCHW) ──► BN
          ──► LayerScale (gamma) ──► DropPath ──► + x  (residual)

    Args:
        dim (int):                  Channel dimension (in == out).
        kernel_size (int):          DW conv kernel. 3/5 = small, 7–17 = large-kernel.
        drop_path (float):          Stochastic depth rate (0 = disabled).
        layer_scale_init_value (float): Layer-scale init (1e-6 default).
        deploy (bool):              Inference-only reparameterised mode.
        use_sync_bn (bool):         Use SyncBN (useful for small batch sizes).
        ffn_factor (int):           FFN hidden-dim multiplier (default 4).
        with_cp (bool):             Gradient checkpointing to save GPU memory.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 13,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        deploy: bool = False,
        use_sync_bn: bool = False,
        ffn_factor: int = 4,
        with_cp: bool = False,
    ):
        super().__init__()
        self.with_cp = with_cp

        # ── Depthwise large-kernel conv ───────────────────────────────────
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(
                dim, kernel_size, deploy=deploy, use_sync_bn=use_sync_bn
            )
        else:
            assert kernel_size in (3, 5), \
                f"kernel_size must be 0, 3, 5, or >=7; got {kernel_size}"
            self.dwconv = _get_conv2d(
                dim, dim, kernel_size, stride=1,
                padding=kernel_size // 2, dilation=1, groups=dim, bias=deploy,
            )

        # ── BN after dw conv ─────────────────────────────────────────────
        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = _get_bn(dim, use_sync_bn)

        # ── Squeeze-and-Excitation ────────────────────────────────────────
        self.se = SEBlock(dim, max(dim // 4, 1))

        # ── FFN: pointwise expand → GELU+GRN → pointwise project ─────────
        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim),
        )
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy),
        )
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW(),
            )
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                _get_bn(dim, use_sync_bn),
            )

        # ── Layer scale ───────────────────────────────────────────────────
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
            if (not deploy)
            and layer_scale_init_value is not None
            and layer_scale_init_value > 0
            else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # ── Forward ──────────────────────────────────────────────────────────
    def _compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_cp and x.requires_grad:
            return checkpoint.checkpoint(lambda t: t + self._compute_residual(t), x)
        return x + self._compute_residual(x)

    # ── Reparameterise (call before switching to deploy mode) ─────────────
    def reparameterize(self):
        """
        Fuse all training-time BN branches into weight tensors.
        After this call the block is equivalent to deploy=True construction.
        """
        # 1. Merge DilatedReparam branches then absorb BN into dw conv
        if hasattr(self.dwconv, "merge_dilated_branches"):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, "running_var"):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, "lk_origin"):
                self.dwconv.lk_origin.weight.data *= (
                    self.norm.weight / std
                ).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                    self.dwconv.lk_origin.bias - self.norm.running_mean
                ) * self.norm.weight / std
            else:
                conv = nn.Conv2d(
                    self.dwconv.in_channels, self.dwconv.out_channels,
                    self.dwconv.kernel_size,
                    padding=self.dwconv.padding,
                    groups=self.dwconv.groups, bias=True,
                )
                conv.weight.data = (
                    self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                )
                conv.bias.data = (
                    self.norm.bias
                    - self.norm.running_mean * self.norm.weight / std
                )
                self.dwconv = conv
            self.norm = nn.Identity()

        # 2. Absorb layer scale into pwconv2 BN
        final_scale = self.gamma.data if self.gamma is not None else 1
        self.gamma = None

        # 3. Absorb GRN bias + BN into pwconv2 linear
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].beta = None
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_proj = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = (
                linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            )
            linear_bias = (0 if linear.bias is None else linear.bias.data) + grn_bias_proj
            new_linear.bias.data = (
                bn.bias + (linear_bias - bn.running_mean) * bn.weight / std
            ) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])


# ============================================================================
# C3_UniRepLKBlock — the YOLO-compatible CSP wrapper
# ============================================================================

class C3_UniRepLKBlock(nn.Module):
    """
    C3-style CSP module with UniRepLKNetBlock as the bottleneck replacement.

    Matches the architecture shown in the RLK-net diagram:

        input ──┬── cv1 ──────────────────────────── cv3 ── output
                └── cv2 ── UniRepLKNetBlock × n ──┘
                           (concat before cv3)

    Where cv1, cv2 are CBS (Conv-BN-SiLU, K=1, S=1) and cv3 is also CBS K=1.

    Args:
        c1 (int):         Input channels.
        c2 (int):         Output channels.
        n (int):          Number of UniRepLKNetBlock repetitions.
        shortcut (bool):  Unused (kept for YAML API parity with C3/C2f).
                          UniRepLKNetBlock always has its own internal residual.
        g (int):          Unused (UniRepLKNetBlock uses depthwise convs).
        e (float):        Channel expansion ratio for hidden dim.
        kernel_size (int):  DW conv kernel for each UniRepLKNetBlock.
                            Recommended: 13 (default, large-kernel, best accuracy)
                                          7  (lighter, still large-kernel)
                                          5  (small-kernel, fewest params)
                                          3  (smallest, ~C3 speed)
        drop_path (float):  DropPath rate for stochastic depth.
        deploy (bool):      Reparameterised inference mode.
        use_sync_bn (bool): SyncBN for small-batch training.
        ffn_factor (int):   FFN expansion ratio inside each block.
        with_cp (bool):     Gradient checkpointing.

    YAML examples:
        - [-1, 3, C3_UniRepLKBlock, [256]]               # all defaults, k=13
        - [-1, 6, C3_UniRepLKBlock, [512, False, 1, 0.5, 13]]
        - [-1, 3, C3_UniRepLKBlock, [256, False, 1, 0.5, 7, 0.0, False]]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,          # kept for API parity, not used
        g: int = 1,                     # kept for API parity, not used
        e: float = 0.5,
        kernel_size: int = 13,
        drop_path: float = 0.0,
        deploy: bool = False,
        use_sync_bn: bool = False,
        ffn_factor: int = 4,
        with_cp: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)                               # hidden channels

        # CBS K=1, S=1 projections (named cv1/cv2/cv3 to match C3 naming)
        self.cv1 = Conv(c1, c_, 1, 1)                  # shortcut/identity branch
        self.cv2 = Conv(c1, c_, 1, 1)                  # bottleneck branch
        self.cv3 = Conv(2 * c_, c2, 1)                 # final merge

        # UniRepLKNetBlock stack (channel-preserving: c_ → c_)
        self.m = nn.Sequential(
            *[
                UniRepLKNetBlock(
                    dim=c_,
                    kernel_size=kernel_size,
                    drop_path=drop_path,
                    deploy=deploy,
                    use_sync_bn=use_sync_bn,
                    ffn_factor=ffn_factor,
                    with_cp=with_cp,
                )
                for _ in range(n)
            ]
        )

        # Store for reparameterisation helper
        self._deploy = deploy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            cv1(x) ──────────────────────────────┐
                                                   Concat → cv3
            cv2(x) ── UniRepLKNetBlock × n ───────┘
        """
        return self.cv3(torch.cat((self.cv1(x), self.m(self.cv2(x))), dim=1))

    def reparameterize(self):
        """
        Fuse all training-time BN branches in every UniRepLKNetBlock.
        Call this once after training is complete, before switching to eval/TensorRT.
        """
        for block in self.m:
            if isinstance(block, UniRepLKNetBlock):
                block.reparameterize()
        self._deploy = True