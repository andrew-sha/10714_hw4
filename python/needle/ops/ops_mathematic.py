"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar


    def gradient(self, out_grad, node):
        dX = mul_scalar(
          power_scalar(node.inputs[0], self.scalar-1), 
          self.scalar
        )
        return dX * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        X, Y = node.inputs
        dX = out_grad * power_scalar(Y, -1)
        dY = out_grad * (divide(negate(X), power_scalar(Y, 2)))
        
        return dX, dY


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes or (-1, -2)

    def compute(self, a):
        n = len(a.shape)
        if n < 2:
            return a

        if self.axes is None:
            i, j = n - 2, n - 1
        else:
            i, j = self.axes
            # normalize negative axes
            if i < 0:
                i += n
            if j < 0:
                j += n

        order = list(range(n))
        order[i], order[j] = order[j], order[i]
        return a.permute(tuple(order))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape)


    def gradient(self, out_grad, node):
        original_shape = node.inputs[0].shape
        padded_shape = ([1] * (len(self.shape) - len(original_shape))) + list(original_shape)
        
        reduced_axes = [dim for dim, size in enumerate(padded_shape) if size == 1 and self.shape[dim] > 1]

        return reshape(summation(out_grad, tuple(reduced_axes)), original_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.sum(self.axes)

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        axes = self.axes

        # Sum over all axes
        if axes is None:
            ones_shape = (1,) * len(x.shape)
            return broadcast_to(reshape(out_grad, ones_shape), x.shape)

        # No reduction (axes == () or [])
        if isinstance(axes, (tuple, list)) and len(axes) == 0:
            return out_grad

        # Normalize to list of axes
        if isinstance(axes, int):
            axes = [axes]
        else:
            axes = list(axes)

        # Normalize negatives and build reshape shape
        axes = [ax % len(x.shape) for ax in axes]
        shape = list(x.shape)
        for ax in axes:
            shape[ax] = 1

        return broadcast_to(reshape(out_grad, tuple(shape)), x.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        X, Y = node.inputs
        dX, dY = matmul(out_grad, transpose(Y)), matmul(transpose(X), out_grad)

        padded_shape_X = ([1] * (len(dX.shape) - len(X.shape))) + list(X.shape)
        padded_shape_Y = ([1] * (len(dY.shape) - len(Y.shape))) + list(Y.shape)
        
        reduced_axes_X = [dim for dim, size in enumerate(padded_shape_X) if size == 1 and dX.shape[dim] > 1]
        reduced_axes_Y = [dim for dim, size in enumerate(padded_shape_Y) if size == 1 and dY.shape[dim] > 1]

        dX_reduced = summation(dX, tuple(reduced_axes_X))
        dY_reduced = summation(dY, tuple(reduced_axes_Y))

        return reshape(dX_reduced, X.shape), reshape(dY_reduced, Y.shape)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return out_grad * -1


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return a.log()

    def gradient(self, out_grad, node):
        return out_grad * power_scalar(node.inputs[0], -1)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return a.exp()

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return a.maximum(0.0)

    def gradient(self, out_grad, node):
        binary = (node.inputs[0].realize_cached_data() > 0)
        return Tensor(out_grad.realize_cached_data() * binary, dtype="float32")


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return a.tanh()

    def gradient(self, out_grad, node):
        # d/dx tanh(x) = 1 - tanh(x)^2
        one_minus_y2 = add_scalar(negate(power_scalar(node, 2)), 1.0)
        return multiply(out_grad, one_minus_y2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # args is a Python tuple of NDArray from MakeTensorTuple.compute
        arrays = list(args)
        assert len(arrays) > 0
        base = arrays[0]
        ndim = len(base.shape)
        axis = self.axis % (ndim + 1)

        # Validate shapes
        for a in arrays:
            assert a.shape == base.shape, "All tensors must have the same shape to stack"

        out_shape = tuple(list(base.shape[:axis]) + [len(arrays)] + list(base.shape[axis:]))
        out = base.make(out_shape, device=base.device)

        # Place each array into the stacked dimension via slicing
        for i, a in enumerate(arrays):
            index = [slice(None)] * axis + [i] + [slice(None)] * (ndim - axis)
            out[tuple(index)] = a

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Return gradients for each input as a TensorTuple by splitting along axis
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # Produce a tuple of NDArray by slicing along axis and removing that dimension
        axis = self.axis % len(A.shape)
        k = A.shape[axis]
        pre = A.shape[:axis]
        post = A.shape[axis + 1 :]
        out = []
        for i in range(k):
            index = [slice(None)] * axis + [i] + [slice(None)] * (len(A.shape) - axis - 1)
            view = A[tuple(index)]
            out.append(view.reshape(tuple(list(pre) + list(post))))
        return tuple(out)

    def gradient(self, out_grad, node):
        # out_grad is a TensorTuple; stack them back along the split axis
        return stack([out_grad[i] for i in range(len(out_grad))], axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


