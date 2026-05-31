import torch
from torchvision.ops import deform_conv2d


input = torch.rand(4, 3, 10, 10)
kh, kw = 3, 3
weight = torch.rand(5, 3, kh, kw)
# offset and mask should have the same spatial size as the output
# of the convolution. In this case, for an input of 10, stride of 1
# and kernel size of 3, without padding, the output size is 8
offset = torch.rand(4, 2 * kh * kw, 8, 8)
mask = torch.rand(4, kh * kw, 8, 8)
out = deform_conv2d(input, offset, weight, mask=mask)
print(out.shape)