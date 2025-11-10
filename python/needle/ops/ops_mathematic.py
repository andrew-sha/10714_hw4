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
        X, Y = node.inputs
        target_device = Y.device
        if out_grad.device != target_device:
            out_grad = Tensor(out_grad.numpy(), device=target_device, requires_grad=False)
        if X.device != target_device:
            X = Tensor(X.numpy(), device=target_device, requires_grad=X.requires_grad)
        if Y.device != target_device:
            Y = Tensor(Y.numpy(), device=target_device, requires_grad=Y.requires_grad)
        
        
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
        axes = self.axes
        # Direct full reduction
        if axes is None:
            return a.sum(None)
        # Empty tuple/list => no-op
        if isinstance(axes, (tuple, list)) and len(axes) == 0:
            return a
        # Normalize to list of axes
        if isinstance(axes, int):
            norm_axes = [axes % len(a.shape)]
        else:
            norm_axes = [ax % len(a.shape) for ax in axes]
        # Reduce along multiple axes by chaining single-axis sums.
        # Sum higher axes first to avoid index shifts after dimension removal.
        out = a
        for ax in sorted(norm_axes, reverse=True):
            out = out.sum(ax)
        return out

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
        target_device = Y.device
        if out_grad.device != target_device:
            out_grad = Tensor(out_grad.numpy(), device=target_device, requires_grad=False)
        if X.device != target_device:
            X = Tensor(X.numpy(), device=target_device, requires_grad=X.requires_grad)
        if Y.device != target_device:
            Y = Tensor(Y.numpy(), device=target_device, requires_grad=Y.requires_grad)

        dX = matmul(out_grad, transpose(Y))
        dY = matmul(transpose(X), out_grad)

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
        # IMPORTANT: Avoid reshape on non-compact views (would assume contiguous strides)
        # Instead, create a strided view that "squeezes" the split axis by dropping it.
        axis = self.axis % len(A.shape)
        k = A.shape[axis]
        pre = A.shape[:axis]
        post = A.shape[axis + 1 :]
        out = []
        for i in range(k):
            index = [slice(None)] * axis + [i] + [slice(None)] * (len(A.shape) - axis - 1)
            view = A[tuple(index)]  # shape: pre + (1,) + post; strides preserved
            # Build squeezed shape/strides by dropping the size-1 axis
            new_shape = tuple(list(pre) + list(post))
            v_strides = list(view.strides)
            # view has len = len(pre) + 1 + len(post); drop the stride at the size-1 axis
            new_strides = tuple(v_strides[:axis] + v_strides[axis + 1 :])
            out.append(view.as_strided(new_shape, new_strides))
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
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        d = int(self.dilation)
        if d == 0 or self.axes is None or len(self.axes) == 0:
            return a

        # Normalize axes: keep only valid dims; map negatives; drop duplicates
        ndim = len(a.shape)
        valid_axes = []
        for ax in self.axes:
            if -ndim <= ax < ndim:
                axn = ax if ax >= 0 else ax + ndim
                if axn not in valid_axes:
                    valid_axes.append(axn)
        if not valid_axes:
            return a

        # Compute output shape: multiply specified axes by (d+1)
        out_shape = list(a.shape)
        for ax in valid_axes:
            out_shape[ax] = out_shape[ax] * (d + 1)

        out = a.make(tuple(out_shape), device=a.device)
        out.fill(0.0)

        # Build slice to place original values at strides (d+1)
        index = []
        for i in range(len(out_shape)):
            if i in valid_axes:
                index.append(slice(0, out_shape[i], d + 1))
            else:
                index.append(slice(0, out_shape[i], 1))

        out[tuple(index)] = a
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        d = int(self.dilation)
        if d == 0 or self.axes is None or len(self.axes) == 0:
            return a
        ndim = len(a.shape)
        valid_axes = []
        for ax in self.axes:
            if -ndim <= ax < ndim:
                axn = ax if ax >= 0 else ax + ndim
                if axn not in valid_axes:
                    valid_axes.append(axn)
        if not valid_axes:
            return a
        # Build slicing view to take every (d+1)-th element along specified axes
        index = []
        for i, size in enumerate(a.shape):
            if i in valid_axes:
                index.append(slice(0, size, d + 1))
            else:
                index.append(slice(0, size, 1))
        return a[tuple(index)]

    def gradient(self, out_grad, node):
        # Gradient inserts zeros between elements on the same axes
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # A: (N, H, W, Cin) NHWC
        # B: (K, K, Cin, Cout)
        N, H, W, Cin = A.shape
        K, K2, Cin_w, Cout = B.shape
        assert K == K2 and Cin == Cin_w, "Kernel must be square and Cin must match"

        s = int(self.stride)
        p = int(self.padding)

        # Pad input on spatial dimensions if needed
        if p > 0:
            A_pad = A.pad(((0, 0), (p, p), (p, p), (0, 0)))
        else:
            A_pad = A

        Hp, Wp = A_pad.shape[1], A_pad.shape[2]
        H_out = (Hp - K) // s + 1
        W_out = (Wp - K) // s + 1

        # im2col: (N*H_out*W_out, K*K*Cin)
        X_col = A.make((N * H_out * W_out, K * K * Cin), device=A.device)

        col = 0
        for i in range(K):
            for j in range(K):
                patch = A_pad[:, i:i + H_out * s:s, j:j + W_out * s:s, :]  # (N, H_out, W_out, Cin)
                patch_flat = patch.compact().reshape((N * H_out * W_out, Cin))
                X_col[:, col * Cin:(col + 1) * Cin] = patch_flat
                col += 1

        W_col = B.compact().reshape((K * K * Cin, Cout))
        Y_col = X_col @ W_col  # (N*H_out*W_out, Cout)
        Y = Y_col.reshape((N, H_out, W_out, Cout))
        return Y

    def gradient(self, out_grad, node):
        # Inputs
        X, W = node.inputs  # Tensors
        N, H, W_in, Cin = X.shape
        K, _, Cin_w, Cout = W.shape
        s = int(self.stride)
        p = int(self.padding)

        # dX: conv of dilated out_grad with flipped and channel-swapped weights
        if s > 1:
            og = dilate(out_grad, axes=(1, 2), dilation=s - 1)
        else:
            og = out_grad

        W_t = transpose(W)              # swap Cin and Cout -> (K,K,Cout,Cin)
        W_flip = flip(W_t, axes=(0, 1)) # flip spatial dims

        pad_in = K - 1 - p
        if pad_in < 0:
            pad_in = 0  # assume no negative padding in our implementation domain
        dX = conv(og, W_flip, stride=1, padding=pad_in)

        # dW: use im2col on NDArray level
        X_nd = X.realize_cached_data()
        og_nd = out_grad.realize_cached_data()

        # Pad input X on spatial dims
        if p > 0:
            Xp = X_nd.pad(((0, 0), (p, p), (p, p), (0, 0)))
        else:
            Xp = X_nd

        Nn, Hp, Wp, Cin2 = Xp.shape
        H_out = (Hp - K) // s + 1
        W_out = (Wp - K) // s + 1

        # Build im2col for Xp
        X_col = X_nd.make((Nn * H_out * W_out, K * K * Cin2), device=Xp.device)
        col = 0
        for i in range(K):
            for j in range(K):
                patch = Xp[:, i:i + H_out * s:s, j:j + W_out * s:s, :]  # (N, H_out, W_out, Cin)
                X_col[:, col * Cin2:(col + 1) * Cin2] = patch.compact().reshape((Nn * H_out * W_out, Cin2))
                col += 1

        G_col = og_nd.reshape((Nn * H_out * W_out, Cout))
        dW_col = (X_col.permute((1, 0)) @ G_col)  # (K*K*Cin, Cout)
        dW_nd = dW_col.reshape((K, K, Cin2, Cout))

        dW = Tensor(dW_nd, dtype="float32")
        return dX, dW


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


