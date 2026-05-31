import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C2f
from ultralytics.nn.modules.conv import Conv, RepConv

class CircleConv(nn.Module):
    def __init__(self, c1, c2, k=5, s=1, freeze=True):
        """Initialize Conv layer with circle weight.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = k // 2

        # Initialize circle weight based on pseudocode
        # out: [k, k]
        r = k // 2
        center = (r, r)
        mat = torch.zeros(k, k)
        for i in range(k):
            for j in range(k):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= r**2:
                    mat[i, j] = 1

        # Convert to pytorch standard
        # out (view): [1, 1, k, k] for a single computation
        # out (repeat): [c2, c1, k, k] repeated for every input
        weight = mat.view(1, 1, k, k).repeat(c2, c1, 1, 1)

        if freeze:
            self.register_buffer("weight", weight)
        else:
            self.weight = nn.Parameter(weight)

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            stride=self.s,
            padding=self.p,
        )


class TriangleConv(nn.Module):
    def __init__(self, c1, c2, k=5, s=1, freeze=True):
        """Initialize Conv layer with triangle weight.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = k // 2

        # Initialize triangle weight
        # out: [k, k]
        height = (k + 1) // 2
        start_row = (k - height) // 2
        center_col = k // 2
        mat = torch.zeros(k, k)

        for i in range(k):
            if start_row <= i < start_row + height:
                half_width = i - start_row

                for j in range(k):
                    if center_col - half_width <= j <= center_col + half_width:
                        mat[i, j] = 1

        # Convert to pytorch standard
        # out (view): [1, 1, k, k] for a single input
        # out (repeat): [c2, c1, k, k] repeated for every input
        weight = mat.view(1, 1, k, k).repeat(c2, c1, 1, 1)

        if freeze:
            self.register_buffer("weight", weight)
        else:
            self.weight = nn.Parameter(weight)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, stride=self.s, padding=self.p)


class EMA(nn.Module):
    """Efficient Multi-Scale Attention Module with Cross-Spatial Learning"""

    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv3x3 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class WeightedConcatN(nn.Module):
    def __init__(self, num_inputs, dimension=1):
        super(WeightedConcatN, self).__init__()
        self.d = dimension
        self.num_inputs = num_inputs
        self.w = nn.Parameter(
            torch.ones(num_inputs, dtype=torch.float32), requires_grad=True
        )
        self.epsilon = 1e-4

    def forward(self, x):
        assert isinstance(x, (list, tuple)), "Input must be a list or tuple of tensors"
        assert len(x) == self.num_inputs, (
            f"Expected {self.num_inputs} inputs, got {len(x)}"
        )

        w = torch.exp(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # Normalize weights
        # Fast normalized fusion
        weighted = [weight[i] * x[i] for i in range(self.num_inputs)]  # Apply weights
        return torch.cat(weighted, dim=self.d)


#### Blocks


class C2f_EMA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions + EMA."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.ema = EMA(self.c)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y1, y2 = self.cv1(x).chunk(2, 1)
        y2 = self.ema(y2)

        y = [y1, y2]

        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y1, y2 = self.cv1(x).split((self.c, self.c), 1)
        y2 = self.ema(y2)
        y = [y1, y2]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)


class ShapeConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, freeze=True):
        """Initialize Shape conv layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
        """
        super().__init__()

        self.triangleconv = TriangleConv(c1, c2, 5, s=s, freeze=freeze)
        self.circleconv = CircleConv(c1, c2, 5, s=s, freeze=freeze)

        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=1, stride=s)
        self.bn1 = nn.BatchNorm2d(c2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c2)
        self.ffm = BiFPNAdd(3)

        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()

    def forward(self, x):

        x1 = self.act1(self.bn1(self.circleconv(x)))
        x2 = self.act2(self.bn2(self.triangleconv(x)))
        x3 = self.act3(self.bn3(self.conv(x)))
        out = self.ffm([x1, x2, x3])

        return out


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h]))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(
            1, 2, 1, 1, 1
        )
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_) * self.scope(x_).sigmoid(), self.scale
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = (
                F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
            )
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


class EFE(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = RepConv(c1, c2, k=k, s=s, act=True)
        self.simam = SimAM()

    def forward(self, x):
        x = self.conv(x)
        out = self.simam(x)
        return out


class EDPIC(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.groups = c1
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(c1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(c1, 1, 1, 1)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        s_x = torch.nn.functional.conv2d(
            x, self.sobel_x.to(x.dtype), padding="same", groups=self.groups
        )
        s_y = torch.nn.functional.conv2d(
            x, self.sobel_y.to(x.dtype), padding="same", groups=self.groups
        )

        s_x_fp32 = s_x.float()
        s_y_fp32 = s_y.float()

        out = torch.sqrt(s_x_fp32**2 + s_y_fp32**2 + 1e-6)
        return out.to(x.dtype)


class AFFM(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.c1 = c1

        self.conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c1, c1)
        self.bn = nn.BatchNorm1d(c1)
        self.act = nn.Sigmoid()

    def forward(self, x: list[torch.Tensor]):

        # [B, 2 * c, H, W]
        _x = torch.cat((x[0], x[1]), dim=1)
        # [B, 2 * c, H, W]
        _x = self.conv(_x)
        _x = self.pool(_x)

        # [B, 2* c]
        _x = torch.flatten(_x, 1)
        _x = self.fc(_x)

        # [B, 2 * c]
        _x = self.act(self.bn(_x))
        _x = _x.unsqueeze(-1).unsqueeze(-1)

        w1, w2 = torch.split(_x, int(self.c1 / 2), dim=1)
        out = x[0] * w1 + x[1] * w2
        return out


class EdgeFEBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()

        self.edpic = EDPIC(c1)
        self.conv = Conv(c1, c2, k, s, p)
        self.shapeconv = ShapeConv(c1, c2, k, s, freeze=True)

    def forward(self, x):
        x1 = self.conv(self.edpic(x))
        x2 = self.shapeconv(x)
        return [x1, x2]


class BiFPNAdd(nn.Module):
    """Weighted feature fusion node (fast normalized fusion from BiFPN paper)."""

    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        # Learnable per-input weights, initialized equally
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights.clone())
        w = w / (w.sum() + self.eps)
        return sum(w[i] * inputs[i] for i in range(len(inputs)))


class WeightedConcatN(nn.Module):
    def __init__(self, num_inputs, dimension=1):
        super(WeightedConcatN, self).__init__()
        self.d = dimension
        self.num_inputs = num_inputs
        self.w = nn.Parameter(
            torch.ones(num_inputs, dtype=torch.float32), requires_grad=True
        )
        self.epsilon = 1e-4

    def forward(self, x):
        assert isinstance(x, (list, tuple)), "Input must be a list or tuple of tensors"
        assert len(x) == self.num_inputs, (
            f"Expected {self.num_inputs} inputs, got {len(x)}"
        )

        w = torch.exp(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # Normalize weights
        # Fast normalized fusion
        weighted = [weight[i] * x[i] for i in range(self.num_inputs)]  # Apply weights
        return torch.cat(weighted, dim=self.d)


# ---------------------------- CSW -----------------------#


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False
        )

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = (
            x.clone()
        )  # !!! Keep the original input intact for the residual connection later
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class Faster_Block(nn.Module):
    def __init__(
        self,
        inc,
        dim,
        n_div=4,
        mlp_ratio=2,
        drop_path=0.1,
        layer_scale_init_value=0.0,
        pconv_fw_type="split_cat",
    ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        )
        return x


class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))


class ConvolutionalGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=hidden_features,
            ),
            act_layer(),
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class Faster_Block_CGLU(nn.Module):
    def __init__(
        self,
        inc,
        dim,
        n_div=4,
        mlp_ratio=2,
        drop_path=0.1,
        layer_scale_init_value=0.0,
        pconv_fw_type="split_cat",
    ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.n_div = n_div

        self.mlp = ConvolutionalGLU(dim)

        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        )
        return x


class C2f_Faster_CGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block_CGLU(self.c, self.c) for _ in range(n))


class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, (3 - 1) // 2),
                groups=dim,
            )
            self.conv0v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=((3 - 1) // 2, 0),
                groups=dim,
            )
            self.conv_spatial_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 2),
                groups=dim,
                dilation=2,
            )
            self.conv_spatial_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(2, 0),
                groups=dim,
                dilation=2,
            )
        elif k_size == 11:
            self.conv0h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, (3 - 1) // 2),
                groups=dim,
            )
            self.conv0v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=((3 - 1) // 2, 0),
                groups=dim,
            )
            self.conv_spatial_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 4),
                groups=dim,
                dilation=2,
            )
            self.conv_spatial_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=(4, 0),
                groups=dim,
                dilation=2,
            )
        elif k_size == 23:
            self.conv0h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, (5 - 1) // 2),
                groups=dim,
            )
            self.conv0v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=((5 - 1) // 2, 0),
                groups=dim,
            )
            self.conv_spatial_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 7),
                stride=(1, 1),
                padding=(0, 9),
                groups=dim,
                dilation=3,
            )
            self.conv_spatial_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(7, 1),
                stride=(1, 1),
                padding=(9, 0),
                groups=dim,
                dilation=3,
            )
        elif k_size == 35:
            self.conv0h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, (5 - 1) // 2),
                groups=dim,
            )
            self.conv0v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=((5 - 1) // 2, 0),
                groups=dim,
            )
            self.conv_spatial_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 11),
                stride=(1, 1),
                padding=(0, 15),
                groups=dim,
                dilation=3,
            )
            self.conv_spatial_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(11, 1),
                stride=(1, 1),
                padding=(15, 0),
                groups=dim,
                dilation=3,
            )
        elif k_size == 41:
            self.conv0h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, (5 - 1) // 2),
                groups=dim,
            )
            self.conv0v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=((5 - 1) // 2, 0),
                groups=dim,
            )
            self.conv_spatial_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 13),
                stride=(1, 1),
                padding=(0, 18),
                groups=dim,
                dilation=3,
            )
            self.conv_spatial_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(13, 1),
                stride=(1, 1),
                padding=(18, 0),
                groups=dim,
                dilation=3,
            )
        elif k_size == 53:
            self.conv0h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, (5 - 1) // 2),
                groups=dim,
            )
            self.conv0v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=((5 - 1) // 2, 0),
                groups=dim,
            )
            self.conv_spatial_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, 17),
                stride=(1, 1),
                padding=(0, 24),
                groups=dim,
                dilation=3,
            )
            self.conv_spatial_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(17, 1),
                stride=(1, 1),
                padding=(24, 0),
                groups=dim,
                dilation=3,
            )

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

class C2f_Unscaled(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
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
