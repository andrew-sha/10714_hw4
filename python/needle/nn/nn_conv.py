"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # Only supports padding="same" (i.e., k//2 on each side)
        self.padding = kernel_size // 2

        # Initialize weights (k, k, in_channels, out_channels) using Kaiming uniform
        w_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        self.weight = Parameter(init.kaiming_uniform(self.in_channels * (self.kernel_size ** 2),
                                                     self.out_channels * (self.kernel_size ** 2),
                                                     shape=w_shape, device=device, dtype=dtype))

        # Optional bias: uniform in +/- 1/sqrt(in_channels * k^2), shape (out_channels,)
        self.bias = None
        if bias:
            bound = 1.0 / (self.in_channels * (self.kernel_size ** 2)) ** 0.5
            self.bias = Parameter(init.rand(self.out_channels, low=-bound, high=bound, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # Input x: NCHW -> convert to NHWC for ops.conv
        # Swap axes (1,2): N C H W -> N H C W
        x = ops.transpose(x, (1, 2))
        # Swap axes (2,3): N H C W -> N H W C
        x_nhwc = ops.transpose(x, (2, 3))

        y = ops.conv(x_nhwc, self.weight, stride=self.stride, padding=self.padding)

        # Add bias if present: bias of shape (Cout,) -> reshape to (1,1,1,Cout) and broadcast
        if self.bias is not None:
            b = ops.reshape(self.bias, (1, self.out_channels))
            b = ops.reshape(b, (1, 1, 1, self.out_channels))
            y = y + ops.broadcast_to(b, y.shape)

        # Convert back to NCHW: NHWC -> NHCW -> NCHW
        y = ops.transpose(y, (2, 3))   # NHWC -> N H C W
        y = ops.transpose(y, (1, 2))   # N H C W -> N C H W
        return y