"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # initialize a weight tensor using Kaiming Uniform
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
    
        # initialize a bias tensor using Kaiming Uniform
        self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype), (1, out_features))) if bias else None


    def forward(self, X: Tensor) -> Tensor:
      product = ops.matmul(X, self.weight)
      return product + ops.broadcast_to(self.bias, product.shape) if self.bias else product


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        B = X.shape[0]
        p = 1
        for n in X.shape[1:]:
          p *= n

        return ops.reshape(X, (B, p))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
      return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.modules:
          x = mod(x)
        
        return x

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        B, C = logits.shape
        y_onehot = init.one_hot(C, y, dtype="float32")

        lse = ops.logsumexp(logits, axes=(1,))                
        picked = ops.summation(y_onehot * logits, axes=(1,))
        loss = ops.summation(lse - picked) * (1.0/B)
        return loss



class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)


    def forward(self, x: Tensor) -> Tensor:
        B, D = x.shape

        if self.training:
          means = ops.summation(x, axes=0) * (1.0/B)

          means_broadcasted = ops.broadcast_to(ops.reshape(means, (1, D)), x.shape)

          ss = ops.summation(ops.power_scalar(x - means_broadcasted, 2.0), axes=0)

          var_biased = ss / B
          std = ops.power_scalar(ops.add_scalar(var_biased, self.eps), .5)

          normed = (x - means_broadcasted) / ops.broadcast_to(ops.reshape(std, (1, D)), x.shape)

          self.running_mean = (1.0-self.momentum) * self.running_mean.data + self.momentum * means.data
          self.running_var = (1.0-self.momentum) * self.running_var.data + self.momentum * var_biased.data

          return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * normed + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        else:
          means_broadcasted = ops.broadcast_to(ops.reshape(self.running_mean, (1, D)), x.shape)
          variances = ops.power_scalar(self.running_var + self.eps, .5)
          variances_broadcasted = ops.broadcast_to(ops.reshape(variances, (1, D)), x.shape)

          normed = (x - means_broadcasted) / variances_broadcasted

          return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * normed + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        B, D = x.shape
        means = ops.reshape(ops.summation(x, axes=1) / self.dim, (B, 1))
        means_broadcasted = ops.broadcast_to(means, x.shape)

        variances = ops.summation(ops.power_scalar(x - means_broadcasted, 2), axes=1) / (self.dim)
        variances = ops.reshape(variances, (B, 1))
        variances = ops.power_scalar(ops.add_scalar(variances, self.eps), .5)
        variances_broadcasted = ops.broadcast_to(variances, x.shape)

        normed = (x - means_broadcasted) / variances_broadcasted

        return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * normed + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
      if self.training:
        mask = init.randb(*x.shape, p=1-self.p, dtype="float32") / (1.0-self.p)
        return x * mask
      else:
        return x


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x