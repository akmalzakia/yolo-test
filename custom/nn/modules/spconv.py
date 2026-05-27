# sp_c2f.py
# Shape-Prior C2f (SP-C2f) — Semantically-Motivated Shape Convolutional C2f
#
# Each bottleneck in the C2f block is initialised with a kernel mask derived
# from the canonical geometry of a real CCTSDB traffic sign category:
#
#   "circle"    → Prohibition signs  (circular, red/black/white)
#   "octagon"   → Prohibition signs  (octagonal stop-sign variant)
#   "triangle"  → Warning signs      (equilateral ▲, yellow/black)
#   "rectangle" → Mandatory signs    (rectangular, blue/white)
#
# Freeze strategy — ES-YOLO layer-based (permanent, no schedule needed):
#   cv1 (first conv)  : shape-prior init, requires_grad=False for entire training
#   cv2 (second conv) : standard random init, freely learned
#
# This directly mirrors ES-YOLO's design:
#   "We fix the kernel parameters in the first shape-prior layer [...].
#    In the second layer the kernel parameters are learned dynamically."
#
# With n=4 bottlenecks, each slot maps 1-to-1 to one sign geometry.
# For n<4 a priority order drops least-common shapes first.
# For n>4 the sequence cycles with a vanilla "none" (plain Bottleneck) fallback.
#
# Integration: call register_sp_c2f() once before constructing any model.
# Reference "SPConvC2f" in YAML backbone entries exactly like "C2f".
#
# Usage:
#   from sp_c2f import register_sp_c2f
#   register_sp_c2f()
#   model = YOLO("sp_yolov8s.yaml")
#   model.train(...)

import copy
import functools

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


# ---------------------------------------------------------------------------
# 1.  Shape mask generators
# ---------------------------------------------------------------------------

def _make_circle_mask(k: int) -> torch.Tensor:
    """
    Filled circle inscribed in a k×k grid.
    Models circular prohibition signs (speed limits, no-entry, etc.)

    Example k=5:
        0 0 1 0 0
        0 1 1 1 0
        1 1 1 1 1
        0 1 1 1 0
        0 0 1 0 0
    """
    r = k // 2
    mask = torch.zeros(k, k)
    for i in range(k):
        for j in range(k):
            if (i - r) ** 2 + (j - r) ** 2 <= r ** 2:
                mask[i, j] = 1.0
    return mask


def _make_octagon_mask(k: int) -> torch.Tensor:
    """
    Regular octagon (chamfered square) in a k×k grid.
    Models octagonal prohibition signs (stop signs, give-way variants).
    Corners are cut by ~(k//4) pixels on each side.

    Example k=5:
        0 1 1 1 0
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        0 1 1 1 0

    Example k=7:
        0 0 1 1 1 0 0
        0 1 1 1 1 1 0
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        0 1 1 1 1 1 0
        0 0 1 1 1 0 0
    """
    mask = torch.ones(k, k)
    cut = max(1, k // 4)          # corner cut size scales with kernel
    for c in range(cut):
        trim = cut - c            # pixels to remove from each side at row c
        for j in range(trim):
            mask[c, j]         = 0.0   # top-left corner
            mask[c, k-1-j]     = 0.0   # top-right corner
            mask[k-1-c, j]     = 0.0   # bottom-left corner
            mask[k-1-c, k-1-j] = 0.0   # bottom-right corner
    return mask


def _make_triangle_mask(k: int) -> torch.Tensor:
    """
    Filled equilateral-ish triangle, apex at top (▲), base at bottom.
    Matches Chinese standard warning sign geometry (GB 5768).
    Uses ES-YOLO filled style: each row is a solid horizontal band.

    Example k=5:
        0 0 1 0 0   ← apex
        0 1 1 1 0
        1 1 1 1 1   ← base
        0 0 0 0 0   (empty rows keep kernel size consistent with others)
        0 0 0 0 0
    """
    r = k // 2
    mask = torch.zeros(k, k)
    # Only fill the top ceil(k*0.6) rows so the base sits in the middle
    active_rows = max(3, round(k * 0.6))
    for i in range(active_rows):
        # row 0 = apex (narrowest), row active_rows-1 = base (widest)
        half_w = round(i * r / (active_rows - 1))
        for j in range(k):
            if j >= r - half_w and j <= r + half_w:
                mask[i, j] = 1.0
    return mask


def _make_rectangle_mask(k: int) -> torch.Tensor:
    """
    Horizontal rectangle occupying the central ~60% of rows.
    Models rectangular mandatory signs (direction plates, lane markings).

    Example k=5:
        0 0 0 0 0
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        0 0 0 0 0

    Example k=7:
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
    """
    mask = torch.zeros(k, k)
    margin = max(1, k // 5)       # blank rows top and bottom
    for i in range(margin, k - margin):
        mask[i, :] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 2.  Shape catalogue & priority sequence
# ---------------------------------------------------------------------------

# Maps shape name → generator function(kernel_size) → (k, k) float tensor
SHAPE_CATALOGUE = {
    "circle":    _make_circle_mask,     # prohibition: circular
    "octagon":   _make_octagon_mask,    # prohibition: octagonal
    "triangle":  _make_triangle_mask,   # warning:     triangular (▲)
    "rectangle": _make_rectangle_mask,  # mandatory:   rectangular
    "none":      None,                  # vanilla random-init bottleneck
}

# Priority order for assigning shapes to n bottleneck slots.
# n=1 → circle only  (most common sign shape)
# n=2 → circle + triangle
# n=3 → circle + triangle + octagon
# n=4 → full semantic set  ← recommended
# n>4 → extras get "none" (standard bottleneck)
SEMANTIC_SHAPE_SEQUENCE = ["circle", "triangle", "octagon", "rectangle"]


def _resolve_shapes(n: int) -> list[str]:
    """Return a list of n shape keys following the semantic priority order."""
    seq = SEMANTIC_SHAPE_SEQUENCE + ["none"] * max(0, n - len(SEMANTIC_SHAPE_SEQUENCE))
    return seq[:n]


def _make_bottleneck(
    c1: int,
    c2: int,
    shortcut: bool,
    g: int,
    kernel_size: int,
    shape: str,
) -> nn.Module:
    """
    Factory that returns the right bottleneck type for a given shape key.

    "none"       → plain Ultralytics Bottleneck  (zero SP overhead)
    anything else → ShapePriorBottleneck
    """
    if shape == "none":
        from ultralytics.nn.modules.block import Bottleneck
        return Bottleneck(
            c1, c2,
            shortcut=shortcut,
            g=g,
            k=(kernel_size, kernel_size),
            e=1.0,
        )
    return ShapePriorBottleneck(
        c1, c2,
        shortcut=shortcut,
        g=g,
        k=(kernel_size, kernel_size),
        e=1.0,
        shape=shape,
    )


# ---------------------------------------------------------------------------
# 3.  ShapePriorBottleneck
# ---------------------------------------------------------------------------

class ShapePriorBottleneck(nn.Module):
    """
    Bottleneck with ES-YOLO layer-based shape-prior freeze strategy.

    cv1 (first conv)  — shape-prior init + requires_grad=False permanently.
        Acts as a fixed geometric feature extractor for the entire training run.
        Weights outside the shape mask are zeroed at init and stay zero.
        BN after cv1 remains trainable — it normalises fixed filter responses.

    cv2 (second conv) — standard random init, fully learnable.
        Refines the shape-biased features from cv1 freely.

    Mirrors ES-YOLO Section 3.3:
        "We fix the kernel parameters in the first shape-prior layer [...].
         In the second layer the kernel parameters are learned dynamically."

    Args:
        c1, c2   : input / output channels
        shortcut : residual add if c1 == c2
        g        : conv groups (cv2 only)
        k        : (cv1_kernel, cv2_kernel) sizes
        e        : hidden channel expansion
        shape    : key in SHAPE_CATALOGUE
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple = (3, 3),
        e: float = 0.5,
        shape: str = "circle",
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.shape = shape

        # ── Apply shape mask to cv1, then permanently freeze its weights ─
        mask_fn  = SHAPE_CATALOGUE[shape]
        raw      = mask_fn(k[0])                                # (k, k)
        w        = self.cv1.conv.weight                         # (C_out, C_in, k, k)
        expanded = raw.unsqueeze(0).unsqueeze(0).expand_as(w).clone()
        self.register_buffer("shape_mask", expanded)

        with torch.no_grad():
            self.cv1.conv.weight.data.mul_(self.shape_mask)

        self.cv1.conv.weight.requires_grad_(False)  # permanent — no schedule needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"shape={self.shape}, "
                f"cv1_frozen={not self.cv1.conv.weight.requires_grad})")


# ---------------------------------------------------------------------------
# 4.  SPConvC2f  — the main block
# ---------------------------------------------------------------------------

class SPConvC2f(nn.Module):
    """
    Shape-Prior C2f with semantically-motivated bottleneck specialisation.

    For n=4 (recommended), the four bottleneck slots map to:
        slot 0 → circle    (prohibition circular signs)
        slot 1 → triangle  (warning triangular signs)
        slot 2 → octagon   (prohibition octagonal signs)
        slot 3 → rectangle (mandatory rectangular signs)

    Each ShapePriorBottleneck has its cv1 permanently frozen (ES-YOLO strategy)
    and cv2 freely learned.  Plain "none" slots are standard Bottlenecks.
    The C2f concat fuses all shape-specific streams — no added parameters.

    Args:
        c1, c2      : input / output channels
        n           : number of bottleneck blocks (4 = full semantic set)
        shortcut    : residual shortcuts in bottlenecks
        g           : groups
        e           : expansion ratio
        shapes      : explicit list of shape keys length n; None → auto
        kernel_size : spatial kernel for shape convs (3 or 5)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        shapes: list = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        if shapes is None:
            shapes = _resolve_shapes(n)
        assert len(shapes) == n, f"shapes length {len(shapes)} != n={n}"

        self.m = nn.ModuleList(
            _make_bottleneck(
                self.c, self.c,
                shortcut=shortcut,
                g=g,
                kernel_size=kernel_size,
                shape=shapes[i],
            )
            for i in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def __repr__(self):
        shapes = [getattr(b, "shape", "none") for b in self.m]
        return (f"{self.__class__.__name__}("
                f"c={self.c}, n={len(self.m)}, shapes={shapes})")


# ---------------------------------------------------------------------------
# 5.  tasks.py integration — monkey-patch, no Ultralytics source edits needed
# ---------------------------------------------------------------------------

def register_sp_c2f():
    """
    Register SPConvC2f into Ultralytics' parse_model.
    Call once before constructing any YOLO model.

    Strategy:
      - Inject SPConvC2f into tasks.py module globals so the YAML string
        "SPConvC2f" resolves through globals()[m].
      - Wrap parse_model: temporarily substitute "SPConvC2f" → "C2f" in the
        config dict so the original parser handles all channel arithmetic and
        depth/width scaling correctly, then swap the built C2f layers back to
        SPConvC2f in-place using the resolved dimensions + original SP args.
    """
    import ultralytics.nn.tasks as _tasks

    if getattr(_tasks, "_sp_c2f_patched", False):
        print("[SP-C2f] Already registered — skipping.")
        return

    # Inject class references so globals()[m] works for YAML resolution
    _tasks.SPConvC2f            = SPConvC2f
    _tasks.ShapePriorBottleneck = ShapePriorBottleneck

    _original_parse_model = _tasks.parse_model

    @functools.wraps(_original_parse_model)
    def _new_parse_model(d, ch, verbose=True):

        d2         = copy.deepcopy(d)
        all_layers = d2.get("backbone", []) + d2.get("head", [])
        sp_entries = []   # list of (flat_index, original_args_copy)

        for idx, row in enumerate(all_layers):
            if row[2] == "SPConvC2f":
                sp_entries.append((idx, list(row[3])))  # save original args
                row[2] = "C2f"                          # temp substitute

        # Let original parser resolve channels, depth/width scaling, n, etc.
        model = _original_parse_model(d2, ch, verbose)

        if not sp_entries:
            return model

        # ── Re-substitute each C2f layer with SPConvC2f ──────────────────
        for seq_idx, orig_args in sp_entries:
            c2f_layer = model[seq_idx]

            # Read resolved dimensions directly from the built C2f layer
            c1_actual = c2f_layer.cv1.conv.in_channels
            c2_actual = c2f_layer.cv2.conv.out_channels
            n_actual  = len(c2f_layer.m)
            shortcut  = (c2f_layer.m[0].add
                         if hasattr(c2f_layer.m[0], "add") else False)

            # SP-specific extra args in YAML: [c2, shortcut, kernel_size]
            # orig_args[0] = c2 (raw, pre-scaling)
            # orig_args[1] = shortcut
            # orig_args[2] = kernel_size  (optional, default 3)
            kernel_size = int(orig_args[2]) if len(orig_args) >= 3 else 3

            new_layer = SPConvC2f(
                c1_actual, c2_actual,
                n=n_actual,
                shortcut=shortcut,
                kernel_size=kernel_size,
            )
            new_layer = new_layer.to(next(c2f_layer.parameters()).device)
            model[seq_idx] = new_layer

        return model

    _tasks.parse_model    = _new_parse_model
    _tasks._sp_c2f_patched = True
    print("[SP-C2f] SPConvC2f registered into ultralytics parse_model ✅")


# ---------------------------------------------------------------------------
# 6.  Future experimentation — epoch-based freeze (not active)
# ---------------------------------------------------------------------------
# An epoch-based curriculum freeze strategy (freeze cv1 for K epochs, then
# release) was considered but not adopted in this implementation.
# ES-YOLO uses permanent layer-based freezing, which is simpler, requires no
# training callbacks, and is the strategy used here.
#
# If you want to experiment with epoch-based unfreezing in future work,
# the approach would be:
#   1. Remove requires_grad_(False) from ShapePriorBottleneck.__init__
#   2. Re-add the weight.data.mul_(mask) re-application in forward()
#      guarded by a self._frozen flag
#   3. Add a set_epoch() method that flips _frozen at a threshold
#   4. Add an SPConvEpochCallback that calls set_epoch() each epoch via
#      model.add_callback("on_train_epoch_start", callback)