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
    Shape-Prior C2f — C2f with semantically specialised bottlenecks.

    Slot assignment for n=4:
        0 → circle (prohibition circular)
        1 → triangle (warning ▲)
        2 → octagon (prohibition octagonal)
        3 → rectangle (mandatory rectangular)
    Slots > 4 → plain Bottleneck.

    YAML args (tasks.py injects c1 and n automatically like C2f):
        [c2, shortcut, g, e, shapes, kernel_size]
    """

    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, shapes=None, kernel_size=3
    ):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import Bottleneck

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        if shapes is None:
            shapes = _resolve_shapes(n)
        assert len(shapes) == n, f"shapes length {len(shapes)} != n={n}"

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
            if s != "none"
            else Bottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=(kernel_size, kernel_size),
                e=1.0,
            )
            for s in shapes
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
        shapes = [getattr(b, "shape", "none") for b in self.m]
        return (
            f"{self.__class__.__name__}(c={self.c}, n={len(self.m)}, shapes={shapes})"
        )


# ---------------------------------------------------------------------------
# Callback — re-apply cv1 freeze after Ultralytics trainer unfreezes it
# ---------------------------------------------------------------------------


class SPConvFreezeCallback:
    """
    Ultralytics on_train_start callback that re-freezes all
    ShapePriorBottleneck cv1 weights after the trainer's setup loop
    unconditionally sets requires_grad=True on every parameter.

    The trainer's freeze loop runs inside _setup_train(), which is called
    just before on_train_start. This callback therefore fires at exactly
    the right moment to restore the permanent cv1 freeze.

    Attach with:
        SPConvFreezeCallback.attach(yolo_model)
    """

    def __init__(self, pytorch_model: nn.Module):
        self.model = pytorch_model

    def __call__(self, trainer):
        n = 0
        for m in self.model.modules():
            if isinstance(m, ShapePriorBottleneck):
                m.cv1.conv.weight.requires_grad_(False)
                n += 1
        if n:
            import logging

            logging.getLogger("ultralytics").info(
                f"[SP-C2f] Re-applied cv1 freeze to {n} ShapePriorBottleneck layers ✅"
            )

    @staticmethod
    def attach(yolo_wrapper) -> "SPConvFreezeCallback":
        """Attach to a ultralytics.YOLO instance and return the callback."""
        cb = SPConvFreezeCallback(yolo_wrapper.model)
        yolo_wrapper.add_callback("on_train_start", cb)
        return cb
