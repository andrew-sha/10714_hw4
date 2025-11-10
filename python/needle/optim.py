"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for idx, param in enumerate(self.params):
          g = param.grad.data + self.weight_decay * param.data
          new_u = self.momentum * self.u.get(param, 0.0*g) + (1.0-self.momentum) * g
          param.data = param.data - self.lr * new_u
          self.u[param] = new_u

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
      self.t += 1
      for param in self.params:
        g = param.grad.data + self.weight_decay * param.data

        u_new = self.beta1 * self.m.get(param, 0.0) + (1.0-self.beta1) * g
        v_new = self.beta2 * self.v.get(param, 0.0) + (1.0-self.beta2) * (g * g)
        u_new_corrected = u_new.data / (1.0-(self.beta1**self.t))
        v_new_corrected = v_new.data / (1.0-(self.beta2**self.t))
        
        param.data = param.data - ((self.lr * u_new_corrected) / (v_new_corrected.data**.5 + self.eps))

        self.m[param] = u_new
        self.v[param] = v_new
