"""
dyconv_c2f.py — Dynamic Convolution C2f with Data-Driven Kernel Initialisation
================================================================================
Install at: ultralytics/nn/dyconv_c2f.py

Architecture
------------
DyConvBottleneck
  cv1 = DynamicConv   ← K-kernel ensemble, input-conditioned gating (CondConv/DyConv style)
  cv2 = standard Conv ← freely learned

DynamicConv internals
  kernels      : nn.Parameter  (K, c_out, c_in, k, k)  — K expert filters
  attention_fc : Linear(c_in → K) + Softmax            — per-sample gate
  ortho_loss() : ||W W^T - I||^2_F                     — diversity regulariser

Initialisation (Strategy 2)
----------------------------
  Pass `proto_crops` (list of (k,k) grayscale numpy arrays, one per kernel slot)
  to `DyConvC2f(..., proto_crops=crops)`. Each crop is broadcast across
  (c_out, c_in) to seed the corresponding expert kernel.

  If `proto_crops=None` (default) → Kaiming uniform init (standard baseline).

Orthogonality Regularisation (Strategy 4)
------------------------------------------
  Call `model.ortho_loss(lambda_ortho)` in your training loop and add it to the
  task loss. The loss penalises off-diagonal entries of the Gram matrix W W^T,
  forcing the K expert kernels to remain diverse throughout training.

  Typical lambda_ortho: 1e-4 to 1e-3. Start at 1e-3 for first 10 epochs, then
  decay to 1e-4 for the remainder (optional schedule).

tasks.py integration
---------------------
  patch_tasks.py adds DyConvC2f to base_modules and repeat_modules.
  YAML args (c1 and n are auto-injected by tasks.py):
    - [-1, 3, DyConvC2f, [128, True, 1, 0.5, 3, 4]]
                                                  ^K  ← number of expert kernels

References
----------
  CondConv : Yang et al., NeurIPS 2019
  DyConv   : Chen et al., CVPR 2020 — Dynamic Convolution: Attention over Kernels
  Ortho reg: Bansal et al., ICLR 2018 — Can We Gain More from Orthogonality?
  K-means init: motivated by data-dependent init (Krähenbühl et al., ICLR 2016)
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kaiming_uniform_init(tensor: torch.Tensor, a: float = math.sqrt(5)) -> None:
    """PyTorch default Conv2d init — kept identical for reproducibility."""
    nn.init.kaiming_uniform_(tensor, a=a)


def _proto_to_kernel(
    proto: np.ndarray,
    c_out: int,
    c_in: int,
    k: int,
    scale: float = 0.1,
) -> torch.Tensor:
    """
    Convert a (k, k) prototype crop into a (c_out, c_in, k, k) kernel tensor.

    The prototype is resized to (k, k) if needed, normalised to [-1, 1], then
    broadcast across all (c_out, c_in) pairs. A small Gaussian noise (std=0.01)
    is added so different channels are not perfectly identical — this prevents
    the ortho regulariser from immediately collapsing gradient norms.

    Args:
        proto : (k, k) float32 numpy array, pixel values in [0, 255] or [0, 1].
        c_out : output channels of the conv layer.
        c_in  : input channels of the conv layer.
        k     : spatial kernel size.
        scale : weight scale applied after normalisation (default 0.1, keeps
                activations in a reasonable range with BN downstream).

    Returns:
        Tensor of shape (c_out, c_in, k, k).
    """
    import cv2  # lazy — only needed at init time
    proto = np.array(proto, dtype=np.float32)

    # Resize if necessary
    if proto.shape != (k, k):
        proto = cv2.resize(proto, (k, k), interpolation=cv2.INTER_LINEAR)

    # Normalise to [-1, 1]
    p_min, p_max = proto.min(), proto.max()
    if p_max > p_min:
        proto = 2.0 * (proto - p_min) / (p_max - p_min) - 1.0
    else:
        proto = np.zeros_like(proto)

    proto_t = torch.from_numpy(proto)  # (k, k)
    # Broadcast to (c_out, c_in, k, k)
    kernel = proto_t.unsqueeze(0).unsqueeze(0).expand(c_out, c_in, k, k).clone()
    kernel = kernel * scale
    # Add tiny noise so channels are not identical
    kernel = kernel + torch.randn_like(kernel) * 0.01
    return kernel


# ---------------------------------------------------------------------------
# k-means prototype extraction
# ---------------------------------------------------------------------------

def extract_prototypes(
    crop_paths: Optional[List[str]] = None,
    crop_arrays: Optional[List[np.ndarray]] = None,
    k_clusters: int = 4,
    kernel_size: int = 3,
    max_crops: int = 2000,
    random_state: int = 42,
) -> List[np.ndarray]:
    """
    Extract K prototype kernels from bounding-box crops via k-means clustering.

    This implements Strategy 2 (data-driven kernel initialisation). Run once
    before training; the returned prototypes are passed to DyConvC2f as
    `proto_crops`.

    Args:
        crop_paths   : list of file paths to sign crop images (any format
                       readable by OpenCV). Either this OR crop_arrays required.
        crop_arrays  : list of numpy arrays (H, W) or (H, W, C), already loaded.
        k_clusters   : number of cluster centroids = number of DyConv kernels K.
        kernel_size  : target spatial size of each prototype (should match the
                       DyConv kernel_size used in the model).
        max_crops    : subsample to this many crops for speed (shuffled randomly).
        random_state : numpy random seed for reproducibility.

    Returns:
        List of K numpy arrays, each of shape (kernel_size, kernel_size),
        dtype float32, values in [0, 255]. These are the k-means centroids and
        directly encode the dominant visual patterns in the training crops.

    Usage example
    -------------
        from dyconv_c2f import extract_prototypes

        protos = extract_prototypes(
            crop_paths=my_crop_list,
            k_clusters=4,
            kernel_size=3,
        )
        # Then pass to model:
        model = YOLO('dyconv_yolov8s.yaml')
        # proto injection happens inside DyConvC2f.__init__ via proto_crops arg
        # See patch_tasks.py for how to thread protos through YAML construction.

    Notes
    -----
    - Crops are converted to grayscale (single-channel) before clustering.
      Grayscale captures shape/intensity structure; colour is not needed.
    - Each crop is resized to (kernel_size, kernel_size) — very small.
      k-means therefore clusters on low-frequency spatial structure, which is
      exactly what a conv kernel captures.
    - With CCTSDB crops the centroids naturally resemble circular blobs,
      triangular wedges, and rectangular bands — matching the sign categories.
      On a different dataset they will cluster around whatever shapes dominate.
    """
    import cv2
    from sklearn.cluster import KMeans  # sklearn for reliable k-means

    rng = np.random.default_rng(random_state)

    # ---- load crops --------------------------------------------------------
    arrays: List[np.ndarray] = []

    if crop_paths is not None:
        for p in crop_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                arrays.append(img)

    if crop_arrays is not None:
        for arr in crop_arrays:
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            arrays.append(arr.astype(np.float32))

    if len(arrays) == 0:
        raise ValueError(
            "extract_prototypes: no valid crops loaded. "
            "Provide either crop_paths or crop_arrays."
        )

    # ---- subsample ---------------------------------------------------------
    if len(arrays) > max_crops:
        idx = rng.choice(len(arrays), size=max_crops, replace=False)
        arrays = [arrays[i] for i in idx]

    # ---- resize to kernel_size x kernel_size and flatten -------------------
    flat: List[np.ndarray] = []
    for arr in arrays:
        resized = cv2.resize(arr.astype(np.float32), (kernel_size, kernel_size),
                             interpolation=cv2.INTER_AREA)
        flat.append(resized.flatten())  # (kernel_size^2,)

    X = np.stack(flat, axis=0)  # (N, kernel_size^2)

    # Normalise rows to [0, 1] for comparable clustering
    row_min = X.min(axis=1, keepdims=True)
    row_max = X.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min > 1e-6, row_max - row_min, 1.0)
    X_norm = (X - row_min) / denom

    # ---- k-means -----------------------------------------------------------
    kmeans = KMeans(
        n_clusters=k_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(X_norm)

    # ---- reshape centroids to (k, k) images --------------------------------
    prototypes: List[np.ndarray] = []
    for centroid in kmeans.cluster_centers_:
        proto = centroid.reshape(kernel_size, kernel_size)
        # Scale back to [0, 255] for _proto_to_kernel compatibility
        proto = (proto * 255.0).astype(np.float32)
        prototypes.append(proto)

    return prototypes


# ---------------------------------------------------------------------------
# DynamicConv — K-kernel ensemble with input-conditioned attention gate
# ---------------------------------------------------------------------------

class DynamicConv(nn.Module):
    """
    Dynamic convolution: weighted sum of K expert conv kernels where weights
    are conditioned on the input via a lightweight attention gate.

    Architecture follows DyConv (Chen et al., CVPR 2020) with:
      - K parallel expert kernels stored as a single (K, c_out, c_in, k, k) Parameter
      - Global Average Pooling → FC(c_in, K) → Softmax(T) attention gate
      - Output: sum_k(alpha_k * conv(x, W_k))

    Orthogonality regularisation (Strategy 4)
    ------------------------------------------
    Call ortho_loss() to obtain the regularisation term. The Gram matrix is
    computed over the flattened kernel vectors (each kernel → c_out * k*k dims),
    so the constraint penalises kernels that span similar directions in weight
    space, forcing specialisation.

    Args:
        c_in      : input channels.
        c_out     : output channels.
        k         : kernel spatial size (default 3).
        stride    : conv stride.
        padding   : conv padding (default k//2 = same).
        K         : number of expert kernels (default 4).
        T         : softmax temperature — higher T → more uniform mixing,
                    lower T → harder selection. Default 30 follows DyConv paper.
        proto_crops: list of K numpy arrays (h, w) for data-driven init.
                    If None → Kaiming uniform init.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        K: int = 4,
        T: float = 30.0,
        proto_crops: Optional[List[np.ndarray]] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.stride = stride
        self.padding = k // 2 if padding is None else padding
        self.K = K
        self.T = T

        # K expert kernels packed as a single parameter — (K, c_out, c_in, k, k)
        self.weight = nn.Parameter(torch.empty(K, c_out, c_in, k, k))

        # Attention gate: GAP → FC → Softmax
        # No bias on FC: output is only relative (goes through softmax anyway)
        self.attention_fc = nn.Linear(c_in, K, bias=True)

        # BN + activation applied after the weighted sum
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

        self._init_weights(proto_crops)

    # ------------------------------------------------------------------ init

    def _init_weights(self, proto_crops: Optional[List[np.ndarray]]) -> None:
        """Initialise expert kernels from prototypes or Kaiming uniform."""
        if proto_crops is not None:
            if len(proto_crops) != self.K:
                raise ValueError(
                    f"DynamicConv: expected {self.K} proto_crops, "
                    f"got {len(proto_crops)}."
                )
            with torch.no_grad():
                for i, proto in enumerate(proto_crops):
                    kernel = _proto_to_kernel(
                        proto, self.c_out, self.c_in, self.k
                    )
                    self.weight[i].copy_(kernel)
        else:
            # Kaiming uniform per expert (same as nn.Conv2d default)
            for i in range(self.K):
                _kaiming_uniform_init(self.weight[i])

        # Zero-init attention FC → uniform mixing at epoch 0
        # This gives a "warm start" where all kernels contribute equally,
        # and specialisation emerges through training.
        nn.init.zeros_(self.attention_fc.weight)
        nn.init.zeros_(self.attention_fc.bias)

    # --------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ---- Attention gate ------------------------------------------------
        # GAP: (B, C, H, W) → (B, C)
        gap = x.mean(dim=[2, 3])
        # FC + softmax with temperature: (B, K)
        alpha = torch.softmax(self.attention_fc(gap) * self.T, dim=1)  # (B, K)

        # ---- Aggregated kernel --------------------------------------------
        # alpha: (B, K) → (B, K, 1, 1, 1, 1) for broadcasting
        alpha_view = alpha.view(B, self.K, 1, 1, 1, 1)
        # weight: (K, c_out, c_in, k, k) — no batch dim
        # Weighted sum over K: (B, c_out, c_in, k, k)
        agg_weight = (alpha_view * self.weight.unsqueeze(0)).sum(dim=1)

        # ---- Per-sample convolution ----------------------------------------
        # Reshape for group conv trick: treat batch as groups.
        # .contiguous() is required because C2f's chunk() produces non-contiguous
        # views, and view() requires contiguous memory layout.
        # x:          (B, C, H, W)   → (1, B*C, H, W)
        # agg_weight: (B, c_out, c_in, k, k) → (B*c_out, c_in, k, k)
        x_reshape = x.contiguous().view(1, B * C, H, W)
        w_reshape = agg_weight.contiguous().view(B * self.c_out, self.c_in, self.k, self.k)

        out = F.conv2d(
            x_reshape,
            w_reshape,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=B,
        )
        # (1, B*c_out, H_out, W_out) → (B, c_out, H_out, W_out)
        _, _, H_out, W_out = out.shape
        out = out.view(B, self.c_out, H_out, W_out)

        return self.act(self.bn(out))

    # ------------------------------------------------------ ortho loss hook

    def ortho_loss(self) -> torch.Tensor:
        """
        Orthogonality regularisation loss (Strategy 4).

        Computes ||W W^T - I||^2_F where W is the (K, D) matrix of flattened
        expert kernels, D = c_out * c_in * k * k.

        Penalises high cosine similarity between any pair of expert kernels,
        forcing the K filters to specialise in diverse directions of feature
        space. This prevents the dynamic ensemble from degenerating into K
        near-identical copies of the dominant mode.

        Returns:
            Scalar tensor (requires_grad=True). Add to task loss:
                loss = task_loss + lambda_ortho * model.ortho_loss()

        Reference:
            Bansal et al. "Can We Gain More from Orthogonality Regularizations
            in Training Deep Neural Networks?" NeurIPS 2018.
        """
        # W: (K, D) where D = c_out * c_in * k * k
        W = self.weight.view(self.K, -1)
        # Gram matrix: (K, K)
        gram = W @ W.T
        # Off-diagonal penalty: ||gram - I||^2_F
        identity = torch.eye(self.K, device=W.device, dtype=W.dtype)
        loss = ((gram - identity) ** 2).sum()
        return loss


# ---------------------------------------------------------------------------
# DyConvBottleneck
# ---------------------------------------------------------------------------

class DyConvBottleneck(nn.Module):
    """
    Bottleneck with DynamicConv as cv1, standard Conv as cv2.

    Mirrors the structure of the standard Ultralytics Bottleneck:
      x → cv1 (DynamicConv) → cv2 (Conv) → [+ shortcut if shapes match]

    The DynamicConv at cv1 provides input-adaptive feature extraction via
    the K-kernel ensemble. cv2 remains a standard conv to keep the per-sample
    batched-conv overhead contained to the first transform only.

    Args:
        c1          : input channels (injected by tasks.py).
        c2          : output channels.
        shortcut    : add residual connection if True (default True).
        g           : conv groups for cv2 (default 1).
        e           : expansion ratio for hidden channels (default 0.5).
        k           : kernel size for cv1 DynamicConv (default 3).
        K           : number of expert kernels in DynamicConv (default 4).
        T           : softmax temperature for attention gate (default 30).
        proto_crops : K prototype arrays for data-driven init (default None).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
        K: int = 4,
        T: float = 30.0,
        proto_crops: Optional[List[np.ndarray]] = None,
    ):
        super().__init__()
        # Lazy import to avoid circular imports with ultralytics
        from ultralytics.nn.modules.conv import Conv  # noqa: PLC0415

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = DynamicConv(
            c_in=c1,
            c_out=c_,
            k=k,
            stride=1,
            K=K,
            T=T,
            proto_crops=proto_crops,
        )
        self.cv2 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ---------------------------------------------------------------------------
# DyConvC2f — drop-in C2f replacement
# ---------------------------------------------------------------------------

class DyConvC2f(nn.Module):
    """
    C2f block with DyConvBottleneck replacing standard Bottleneck.

    Identical external interface to Ultralytics C2f:
      DyConvC2f(c1, c2, n, shortcut, g, e, k, K, T, proto_crops)

    Channel flow (same as C2f):
      x → split(cv1(x)) → [Bottleneck × n] → concat → cv2 → out

    tasks.py YAML after patching (c1 and n are auto-injected):
      - [-1, 3, DyConvC2f, [128, True, 1, 0.5, 3, 4]]
                                                    ^K (number of expert kernels)

    Args:
        c1          : input channels.
        c2          : output channels.
        n           : number of DyConvBottleneck blocks (depth-scaled by tasks.py).
        shortcut    : residual connection in each bottleneck.
        g           : groups for cv2 inside each bottleneck.
        e           : channel expansion ratio (default 0.5).
        k           : DynamicConv kernel size (default 3).
        K           : number of expert kernels per bottleneck (default 4).
        T           : softmax temperature (default 30).
        proto_crops : K prototype arrays. If provided, ALL bottlenecks in this
                      C2f block share the same prototypes for cv1 init.
                      Pass None to use Kaiming init (baseline / ablation).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
        K: int = 4,
        T: float = 30.0,
        proto_crops: Optional[List[np.ndarray]] = None,
    ):
        super().__init__()
        # Lazy imports
        from ultralytics.nn.modules.conv import Conv  # noqa: PLC0415

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            DyConvBottleneck(
                c1=self.c,
                c2=self.c,
                shortcut=shortcut,
                g=g,
                e=1.0,  # no further expansion inside bottleneck
                k=k,
                K=K,
                T=T,
                proto_crops=proto_crops,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split cv1 output into two halves along channel dim
        y = list(self.cv1(x).chunk(2, 1))
        # Sequentially process through bottlenecks, accumulating all outputs
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def ortho_loss(self) -> torch.Tensor:
        """
        Aggregate orthogonality loss across all DyConvBottlenecks in this block.

        Returns sum of ortho losses from every bottleneck's cv1 DynamicConv.
        Use this in the training loop:

            total_ortho = sum(
                m.ortho_loss()
                for m in model.modules()
                if isinstance(m, DyConvC2f)
            )
            loss = task_loss + lambda_ortho * total_ortho
        """
        return sum(bottleneck.cv1.ortho_loss() for bottleneck in self.m)


# ---------------------------------------------------------------------------
# Utility: collect ortho loss from all DyConvC2f blocks in a model
# ---------------------------------------------------------------------------

def collect_ortho_loss(model: nn.Module) -> torch.Tensor:
    """
    Walk all modules in `model` and sum ortho_loss() from every DyConvC2f.

    Usage in training loop (train.py):
        from dyconv_c2f import collect_ortho_loss

        lambda_ortho = 1e-3  # tune; typical range 1e-4 to 1e-3
        loss += lambda_ortho * collect_ortho_loss(model)

    Returns:
        Scalar tensor on the same device as the model. Returns 0.0 tensor
        (with grad) if no DyConvC2f blocks are found.
    """
    total = None
    for module in model.modules():
        if isinstance(module, DyConvC2f):
            l = module.ortho_loss()
            total = l if total is None else total + l
    if total is None:
        # Fallback: return a zero tensor on whatever device the model is on
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        total = torch.tensor(0.0, device=device, requires_grad=True)
    return total


# ---------------------------------------------------------------------------
# OrthoLossScheduler — optional lambda annealing
# ---------------------------------------------------------------------------

class OrthoLossScheduler:
    """
    Linearly decay lambda_ortho from `start` to `end` over `warmup_epochs`.

    After warmup_epochs, lambda_ortho stays at `end` for the rest of training.
    This allows strong geometric diversity pressure early in training (when
    kernels are still near their initialisation and might collapse) and relaxed
    pressure later (when the task loss dominates).

    Usage:
        scheduler = OrthoLossScheduler(start=1e-3, end=1e-4, warmup_epochs=20)

        for epoch in range(total_epochs):
            lambda_ortho = scheduler.get(epoch)
            ...
            loss += lambda_ortho * collect_ortho_loss(model)

    Args:
        start         : initial lambda value (epoch 0).
        end           : final lambda value (after warmup_epochs).
        warmup_epochs : number of epochs over which to decay.
    """

    def __init__(
        self,
        start: float = 1e-3,
        end: float = 1e-4,
        warmup_epochs: int = 20,
    ):
        self.start = start
        self.end = end
        self.warmup_epochs = warmup_epochs

    def get(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.end
        t = min(epoch / self.warmup_epochs, 1.0)
        return self.start + t * (self.end - self.start)