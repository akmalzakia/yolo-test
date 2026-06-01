import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

# region Masks


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

# endregion


# region Bottlenecks
def _resolve_shapes(n):
    return (
        SEMANTIC_SHAPE_SEQUENCE + ["none"] * max(0, n - len(SEMANTIC_SHAPE_SEQUENCE))
    )[:n]


class ShapePriorBottleneck(nn.Module):
    """
    Bottleneck with ES-YOLO layer-based shape-prior freeze.

    cv1 — shape-prior init, requires_grad=False (permanent).

    cv2 — standard random init, freely learned.

    BN after cv1 stays trainable (normalises fixed filter responses).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, shape="circle"):
        super().__init__()

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


NUM_SHAPES = len(SEMANTIC_SHAPE_SEQUENCE)


def _make_grouped_shape_mask(c_out, k):
    """
    Build a (c_out, 1, k, k) mask that assigns each output channel a shape
    prior from SEMANTIC_SHAPE_SEQUENCE in round-robin order.

    With c_out=64 and 4 shapes:
        ch  0-15  -> circle    mask
        ch 16-31  -> triangle  mask
        ch 32-47  -> octagon   mask
        ch 48-63  -> rectangle mask

    For c_out not divisible by 4 the remainder channels get the first shape
    (circle) so the most common sign type gets slightly more capacity.

    Returns: FloatTensor (c_out, 1, k, k)
    """
    masks = [SHAPE_CATALOGUE[s](k) for s in SEMANTIC_SHAPE_SEQUENCE]  # 4 × (k,k)
    base = c_out // NUM_SHAPES
    rem = c_out % NUM_SHAPES

    per_shape = [base + (1 if i < rem else 0) for i in range(NUM_SHAPES)]

    rows = []
    for shape_idx, count in enumerate(per_shape):
        m = masks[shape_idx].unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        rows.append(m.expand(count, 1, k, k))

    return torch.cat(rows, dim=0).clone()  # (c_out, 1, k, k)


class MultiShapeBottleneck(nn.Module):
    """
    Bottleneck with ES-YOLO layer-based shape-prior freeze.

    cv1 — Frozen grouped shape mask.

    cv2 — standard random init, freely learned.

    BN after cv1 stays trainable (normalises fixed filter responses).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

        grouped = _make_grouped_shape_mask(c_, k[0])  # (c_, 1, k, k)
        w = self.cv1.conv.weight  # (c_, c1, k, k)
        mask = grouped.expand_as(w).clone()
        self.register_buffer("shape_mask", mask)

        with torch.no_grad():
            self.cv1.conv.weight.data.mul_(self.shape_mask)
        self.cv1.conv.weight.requires_grad_(False)

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class DualBranchBottleneck(nn.Module):
    """
    Bottleneck with ES-YOLO layer-based shape-prior freeze + MSD-YOLO conv spatial anchor

    cv1 — Frozen grouped shape mask and free learnable conv layer (distributed using c * free_ratio).

    cv2 — standard random init, freely learned.

    BN after cv1 stays trainable (normalises fixed filter responses).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, free_ratio=0.5):
        super().__init__()

        c_ = int(c2 * e)
        free_ratio = max(0.0, min(free_ratio, 0.99))
        c_free = max(1, int(c_ * free_ratio)) if free_ratio > 0 else 0
        c_shape = c_ - c_free

        self.c_shape = c_shape
        self.c_free = c_free

        self.cv1_shape = Conv(c1, c_shape, k[0], 1)
        self.cv1_free = Conv(c1, c_free, k[0], 1) if c_free > 0 else None

        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

        grouped = _make_grouped_shape_mask(c_shape, k[0])  # (c_shape,1,k,k)
        w = self.cv1_shape.conv.weight  # (c_shape,c1,k,k)
        mask = grouped.expand_as(w).clone()
        self.register_buffer("shape_mask", mask)

        with torch.no_grad():
            self.cv1_shape.conv.weight.data.mul_(self.shape_mask)
        self.cv1_shape.conv.weight.requires_grad_(False)

    def forward(self, x):
        shape_feat = self.cv1_shape(x)  # (B, c_shape, H, W)
        if self.cv1_free is not None:
            free_feat = self.cv1_free(x)  # (B, c_free, H, W)
            fused = torch.cat([shape_feat, free_feat], dim=1)
        else:
            fused = shape_feat
        out = self.cv2(fused)
        return x + out if self.add else out


# endregion


class SPConvC2f(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        bottleneck="Multi",
        kernel_size=5,
        free_ratio=0.5,
        g=1,
        e=0.5,
    ):
        super().__init__()

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        if bottleneck == "Multi":
            self.m = nn.ModuleList(
                MultiShapeBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    k=(kernel_size, kernel_size),
                    e=1.0,
                )
                for _ in range(n)
            )
        elif bottleneck == "Dual":
            self.m = nn.ModuleList(
                DualBranchBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    k=(kernel_size, kernel_size),
                    e=1.0,
                    free_ratio=free_ratio,
                )
                for _ in range(n)
            )
        else:
            shapes = _resolve_shapes(n)
            self.m = nn.ModuleList(
                ShapePriorBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    k=(kernel_size, kernel_size),
                    e=1.0,
                    shape=s,
                )
                for s in shapes
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPConvFreezeCallback:
    def __init__(self, pytorch_model: nn.Module):
        self.model = pytorch_model
 
    def __call__(self, trainer):
        n = 0
        for m in self.model.modules():
            if isinstance(m, DualBranchBottleneck):
                m.cv1_shape.conv.weight.requires_grad_(False)
                n += 1
            elif isinstance(m, (MultiShapeBottleneck, ShapePriorBottleneck)):
                m.cv1.conv.weight.requires_grad_(False)
                n += 1
        if n:
            import logging
            logging.getLogger("ultralytics").info(
                f"[SP-C2f] Re-applied cv1 freeze to {n} shape-prior layers ✅"
            )
 
    @staticmethod
    def attach(yolo_wrapper) -> "SPConvFreezeCallback":
        cb = SPConvFreezeCallback(yolo_wrapper.model)
        yolo_wrapper.add_callback("on_train_start", cb)
        return cb
