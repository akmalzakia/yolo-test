# sp_c2f.py  (installed at ultralytics/nn/sp_c2f.py)
# Shape-Prior C2f — all ultralytics imports are lazy (inside methods)
# to avoid circular import with tasks.py.

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1.  Shape mask generators
# ---------------------------------------------------------------------------


def _make_circle_mask(k):
    r = k // 2
    mask = torch.zeros(k, k)
    for i in range(k):
        for j in range(k):
            if (i - r) ** 2 + (j - r) ** 2 <= r**2:
                mask[i, j] = 1.0
    return mask


def _make_octagon_mask(k):
    mask = torch.ones(k, k)
    cut = max(1, k // 4)
    for c in range(cut):
        trim = cut - c
        for j in range(trim):
            mask[c, j] = 0.0
            mask[c, k - 1 - j] = 0.0
            mask[k - 1 - c, j] = 0.0
            mask[k - 1 - c, k - 1 - j] = 0.0
    return mask


def _make_triangle_mask(k):
    r = k // 2
    mask = torch.zeros(k, k)
    active_rows = max(3, round(k * 0.6))
    for i in range(active_rows):
        half_w = round(i * r / (active_rows - 1))
        for j in range(k):
            if j >= r - half_w and j <= r + half_w:
                mask[i, j] = 1.0
    return mask


def _make_rectangle_mask(k):
    mask = torch.zeros(k, k)
    margin = max(1, k // 5)
    for i in range(margin, k - margin):
        mask[i, :] = 1.0
    return mask


SHAPE_CATALOGUE = {
    "circle": _make_circle_mask,
    "triangle": _make_triangle_mask,
    "octagon": _make_octagon_mask,
    "rectangle": _make_rectangle_mask,
}

SEMANTIC_SHAPE_SEQUENCE = ["circle", "triangle", "octagon", "rectangle"]


def _resolve_shapes(n):
    return (
        SEMANTIC_SHAPE_SEQUENCE + ["none"] * max(0, n - len(SEMANTIC_SHAPE_SEQUENCE))
    )[:n]


# ---------------------------------------------------------------------------
# 2.  ShapePriorBottleneck
# ---------------------------------------------------------------------------


class ShapePriorBottleneck(nn.Module):
    """
    Bottleneck with ES-YOLO layer-based shape-prior freeze.

    cv1 — shape-prior init, requires_grad=False (permanent).
    cv2 — standard random init, freely learned.
    BN after cv1 stays trainable (normalises fixed filter responses).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, shape="circle"):
        super().__init__()
        # Lazy import — Conv is only resolved at instantiation time,
        # after tasks.py (and therefore ultralytics.nn) is fully loaded.
        from ultralytics.nn.modules.conv import Conv

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.shape = shape

        mask_fn = SHAPE_CATALOGUE[shape]
        raw = mask_fn(k[0])
        w = self.cv1.conv.weight
        expanded = raw.unsqueeze(0).unsqueeze(0).expand_as(w).clone()
        self.register_buffer("shape_mask", expanded)

        with torch.no_grad():
            self.cv1.conv.weight.data.mul_(self.shape_mask)
        self.cv1.conv.weight.requires_grad_(False)

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"cv1_frozen={not self.cv1.conv.weight.requires_grad})"
        )


# ---------------------------------------------------------------------------
# 3.  SPConvC2f
# ---------------------------------------------------------------------------


class SPConvC2f(nn.Module):
    """
    Shape-Prior C2f with scale-invariant multi-shape bottlenecks.
 
    Each bottleneck is a MultiShapeBottleneck — its cv1 output channels are
    partitioned across all 4 traffic-sign shape priors simultaneously.
    This means full semantic coverage is preserved even when depth scaling
    collapses the block to n=1 on smaller models.
 
    For n>1, every bottleneck has the same grouped-shape init, giving the
    network multiple independent shape-feature streams to compose.
 
    Structurally identical to C2f — cv1/cv2/m/forward — so it integrates
    into tasks.py base_modules and repeat_modules without modification.
 
    YAML args (tasks.py injects c1 and depth-scales n automatically):
        [c2, shortcut, g, e, kernel_size]
        c2          : output channels (auto-scaled by width_multiple)
        shortcut    : bool
        g           : groups for cv2 (default 1)
        e           : hidden expansion ratio (default 0.5)
        kernel_size : spatial kernel for shape convs (default 3)
    """
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=3):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
 
        self.c   = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
 
        self.m = nn.ModuleList(
            MultiShapeBottleneck(
                self.c, self.c,
                shortcut=shortcut, g=g,
                k=(kernel_size, kernel_size), e=1.0,
            )
            for _ in range(n)
        )
 
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"c={self.c}, n={len(self.m)}, "
                f"shapes={SEMANTIC_SHAPE_SEQUENCE} [grouped per bottleneck])")

NUM_SHAPES = len(SEMANTIC_SHAPE_SEQUENCE)   # 4

def _make_grouped_shape_mask(c_out, k):
    """
    Build a (c_out, 1, k, k) mask that assigns each output channel a shape
    prior from SEMANTIC_SHAPE_SEQUENCE in round-robin order.
 
    With c_out=64 and 4 shapes:
        ch  0-15  → circle    mask
        ch 16-31  → triangle  mask
        ch 32-47  → octagon   mask
        ch 48-63  → rectangle mask
 
    For c_out not divisible by 4 the remainder channels get the first shape
    (circle) so the most common sign type gets slightly more capacity.
 
    Returns: FloatTensor (c_out, 1, k, k)
    """
    masks = [SHAPE_CATALOGUE[s](k) for s in SEMANTIC_SHAPE_SEQUENCE]  # 4 × (k,k)
    base  = c_out // NUM_SHAPES
    rem   = c_out  % NUM_SHAPES
 
    per_shape = [base + (1 if i < rem else 0) for i in range(NUM_SHAPES)]
 
    rows = []
    for shape_idx, count in enumerate(per_shape):
        m = masks[shape_idx].unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        rows.append(m.expand(count, 1, k, k))
 
    return torch.cat(rows, dim=0).clone()   # (c_out, 1, k, k)


class MultiShapeBottleneck(nn.Module):
    """
    Bottleneck whose cv1 output channels are partitioned into 4 equal groups,
    each initialised with a different traffic-sign shape mask:
 
        ch 0   … C/4-1  → circle    (prohibition circular)
        ch C/4 … C/2-1  → triangle  (warning ▲)
        ch C/2 … 3C/4-1 → octagon   (prohibition octagonal)
        ch 3C/4 … C-1   → rectangle (mandatory rectangular)
 
    This means even n=1 (smallest scaled model) retains full shape coverage.
    cv1 weights are permanently frozen (ES-YOLO strategy).
    cv2 weights are freely learned — they learn to compose the shape responses.
    BN after cv1 stays trainable.
 
    Args:
        c1, c2   : input / output channels
        shortcut : residual add when c1 == c2
        g        : groups for cv2
        k        : (cv1_k, cv2_k) kernel sizes
        e        : hidden channel expansion ratio
    """
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1  = Conv(c1, c_, k[0], 1)
        self.cv2  = Conv(c_, c2, k[1], 1, g=g)
        self.add  = shortcut and c1 == c2
 
        # Build (c_, 1, k, k) grouped mask and broadcast to (c_, c1, k, k)
        grouped = _make_grouped_shape_mask(c_, k[0])             # (c_, 1, k, k)
        w       = self.cv1.conv.weight                           # (c_, c1, k, k)
        mask    = grouped.expand_as(w).clone()
        self.register_buffer("shape_mask", mask)
 
        with torch.no_grad():
            self.cv1.conv.weight.data.mul_(self.shape_mask)
        self.cv1.conv.weight.requires_grad_(False)
 
    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out
 
    def __repr__(self):
        c_ = self.cv1.conv.out_channels
        base = c_ // NUM_SHAPES
        return (f"{self.__class__.__name__}("
                f"c_hidden={c_}, "
                f"shapes={SEMANTIC_SHAPE_SEQUENCE}, "
                f"ch_per_shape~{base}, "
                f"cv1_frozen={not self.cv1.conv.weight.requires_grad})")


# ---------------------------------------------------------------------------
# Callback — re-apply cv1 freeze after Ultralytics trainer unfreezes it
# ---------------------------------------------------------------------------


class SPConvFreezeCallback:
    """
    on_train_start callback that re-freezes all MultiShapeBottleneck and
    ShapePriorBottleneck cv1 weights after the trainer's _setup_train() loop
    unconditionally sets requires_grad=True on every parameter.
 
    Attach with:
        SPConvFreezeCallback.attach(yolo_model)
    """
 
    def __init__(self, pytorch_model: nn.Module):
        self.model = pytorch_model
 
    def __call__(self, trainer):
        n = 0
        for m in self.model.modules():
            if isinstance(m, (MultiShapeBottleneck, ShapePriorBottleneck)):
                m.cv1.conv.weight.requires_grad_(False)
                n += 1
        if n:
            import logging
            logging.getLogger("ultralytics").info(
                f"[SP-C2f] Re-applied cv1 freeze to {n} shape-prior bottleneck layers ✅"
            )
 
    @staticmethod
    def attach(yolo_wrapper) -> "SPConvFreezeCallback":
        cb = SPConvFreezeCallback(yolo_wrapper.model)
        yolo_wrapper.add_callback("on_train_start", cb)
        return cb

