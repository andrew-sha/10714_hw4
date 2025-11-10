import math
from .init_basic import *
from typing import Any
#from needle.autograd import Tensor  # for type hints


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    # Glorot/Xavier uniform: U(-a, a) with a = gain * sqrt(6 / (fan_in + fan_out))
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", "float32")
    requires_grad = kwargs.get("requires_grad", True)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, device=device, dtype=dtype, requires_grad=requires_grad)


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    # Glorot/Xavier normal: N(0, gain * sqrt(2 / (fan_in + fan_out)))
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", "float32")
    requires_grad = kwargs.get("requires_grad", True)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, device=device, dtype=dtype, requires_grad=requires_grad)

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    # For linear layers when shape not provided
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", "float32")
    requires_grad = kwargs.get("requires_grad", True)
    bound = math.sqrt(6.0 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=requires_grad)


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", "float32")
    requires_grad = kwargs.get("requires_grad", True)

    def _fans_from_shape(shp):
        if shp is None:
            return fan_in, fan_out
        if len(shp) == 2:
            # Linear weights, assume (in_features, out_features)
            return shp[0], shp[1]
        else:
            # Convolutional weights: (..., in_channels, out_channels)
            receptive = 1
            for d in shp[:-2]:
                receptive *= d
            return shp[-2] * receptive, shp[-1] * receptive

    if shape is None:
        fin, fout = fan_in, fan_out
        bound = math.sqrt(6.0 / max(1, fin))
        return rand(fin, fout, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=requires_grad)

    fin, fout = _fans_from_shape(shape)
    bound = math.sqrt(6.0 / max(1, fin))
    return rand(*shape, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=requires_grad)

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", "float32")
    requires_grad = kwargs.get("requires_grad", True)
    std = math.sqrt(2.0 / fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, device=device, dtype=dtype, requires_grad=requires_grad)