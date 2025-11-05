from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        if self.axes: 
          broadcast_shape = tuple([1 if i in self.axes else Z.shape[i] for i in range(len(Z.shape))])
        else:
          broadcast_shape = tuple([1 for _ in range(len(Z.shape))])

        maxima = array_api.max(Z, self.axes)
        
        diffs = Z - array_api.broadcast_to(array_api.reshape(maxima, broadcast_shape), Z.shape)
        diffs = array_api.exp(diffs)

        sums = array_api.sum(diffs, self.axes)
        sums = array_api.log(sums)
        
        return sums + maxima

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]

        if self.axes: 
          broadcast_shape = tuple([1 if i in self.axes else Z.shape[i] for i in range(len(Z.shape))])
        else:
          broadcast_shape = tuple([1 for _ in range(len(Z.shape))])

        maxima = Tensor(array_api.reshape(array_api.max(Z.realize_cached_data(), self.axes), broadcast_shape), dtype="float32")
        diffs = Z - maxima
        diffs = exp(diffs)
        sums = reshape(summation(diffs, self.axes), broadcast_shape)
        P = diffs / broadcast_to(sums, Z.shape)
        return broadcast_to(reshape(out_grad, broadcast_shape), Z.shape) * P

# class LogSumExp(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None) -> None:
#         self.axes = axes

#     @staticmethod
#     def _norm_axes(axes, ndim):
#         if axes is None:
#             return None
#         if isinstance(axes, int):
#             axes = (axes,)
#         else:
#             axes = tuple(axes)
#         return tuple(ax % ndim for ax in axes)

#     def compute(self, Z: NDArray) -> NDArray:
#         axes = self._norm_axes(self.axes, len(Z.shape))
#         if isinstance(axes, tuple) and len(axes) == 0:
#             return Z

#         bshape = (1,) * len(Z.shape) if axes is None else tuple(
#             1 if i in axes else Z.shape[i] for i in range(len(Z.shape))
#         )

#         m = array_api.max(Z, axes)
#         Zm = Z - array_api.broadcast_to(array_api.reshape(m, bshape), Z.shape)
#         expZ = array_api.exp(Zm)
#         s = array_api.sum(expZ, axes)
#         return array_api.log(s) + m

#     def gradient(self, out_grad: Tensor, node: Tensor):
#         Z = node.inputs[0]
#         ndim = len(Z.shape)
#         axes = self._norm_axes(self.axes, ndim)
#         if isinstance(axes, tuple) and len(axes) == 0:
#             return out_grad

#         bshape = (1,) * ndim if axes is None else tuple(
#             1 if i in axes else Z.shape[i] for i in range(ndim)
#         )

#         # stable softmax weights P along reduction axes
#         m = Tensor(
#             array_api.reshape(array_api.max(Z.realize_cached_data(), axes), bshape),
#             dtype="float32",
#         )
#         Zm = Z - m
#         expZ = exp(Zm)
#         s = reshape(summation(expZ, axes), bshape)
#         P = expZ / broadcast_to(s, Z.shape)

#         return broadcast_to(reshape(out_grad, bshape), Z.shape) * P


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)