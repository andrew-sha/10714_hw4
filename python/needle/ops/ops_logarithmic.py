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
        # Normalize axes: None | int | tuple/list[int]
        ndim = len(Z.shape)
        axes = self.axes
        if axes is None:
            norm_axes = None
        elif isinstance(axes, int):
            norm_axes = (axes % ndim,)
        else:
            norm_axes = tuple(ax % ndim for ax in axes)

        # No reduction case
        if isinstance(norm_axes, tuple) and len(norm_axes) == 0:
            return Z

        bshape = (1,) * ndim if norm_axes is None else tuple(
            1 if i in norm_axes else Z.shape[i] for i in range(ndim)
        )

        maxima = Z.max(norm_axes)
        diffs = Z - array_api.broadcast_to(array_api.reshape(maxima, bshape), Z.shape)
        diffs = array_api.exp(diffs)

        sums = array_api.sum(diffs, norm_axes)
        sums = array_api.log(sums)
        return sums + maxima

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        ndim = len(Z.shape)
        axes = self.axes
        if axes is None:
            norm_axes = None
        elif isinstance(axes, int):
            norm_axes = (axes % ndim,)
        else:
            norm_axes = tuple(ax % ndim for ax in axes)

        # No reduction case
        if isinstance(norm_axes, tuple) and len(norm_axes) == 0:
            return out_grad

        bshape = (1,) * ndim if norm_axes is None else tuple(
            1 if i in norm_axes else Z.shape[i] for i in range(ndim)
        )

        # Stable softmax weights along reduction axes
        m_nd = Z.realize_cached_data().max(norm_axes)
        m = Tensor(array_api.reshape(m_nd, bshape), dtype="float32", device=Z.device)
        # Explicitly broadcast to match NDArray backend which doesn't auto-broadcast in ewise ops
        diffs = Z - broadcast_to(m, Z.shape)
        expZ = exp(diffs)
        s = reshape(summation(expZ, norm_axes), bshape)
        P = expZ / broadcast_to(s, Z.shape)
        return broadcast_to(reshape(out_grad, bshape), Z.shape) * P




def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)