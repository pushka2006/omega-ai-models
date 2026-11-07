"""OMEGA NEURAL NETWORK ENGINE: Educational Autograd Engine.

Pure Python/NumPy implementation of automatic differentiation and backpropagation
on computational graphs. Designed for mathematical verification and gradient comparison
against PyTorch autograd.
"""

import math
from typing import Any, Dict, Set, Tuple, Union
import numpy as np
import torch


def _unbroadcast(grad: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """Sum gradient across broadcasted dimensions to match target parameter shape."""
    if grad.shape == target_shape:
        return grad
    if len(target_shape) == 0:
        return np.array(np.sum(grad))
    # Sum leading dimensions
    lead_dims = len(grad.shape) - len(target_shape)
    if lead_dims > 0:
        grad = np.sum(grad, axis=tuple(range(lead_dims)))
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = np.sum(grad, axis=i, keepdims=True)
    return grad.reshape(target_shape)


class EducationalTensor:
    """Scalar/Tensor node tracking computational graphs and gradients via the chain rule."""

    def __init__(self, data: Union[float, int, np.ndarray], _children: Tuple = (), _op: str = "", label: str = ""):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"EducationalTensor(shape={self.data.shape}, op='{self._op}')"

    def __add__(self, other):
        other = other if isinstance(other, EducationalTensor) else EducationalTensor(other)
        out = EducationalTensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, EducationalTensor) else EducationalTensor(other)
        out = EducationalTensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, EducationalTensor) else EducationalTensor(other)
        out = EducationalTensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = EducationalTensor(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30, 30)))
        out = EducationalTensor(s, (self,), "Sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def mse_loss(self, target: np.ndarray):
        diff = self.data - target
        loss_val = np.mean(diff ** 2)
        out = EducationalTensor(loss_val, (self,), "MSE")

        def _backward():
            self.grad += (2.0 / self.data.size) * diff * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Topological sort and backpropagation across the computational graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


def compare_gradients_with_pytorch(
    x_val: np.ndarray, w_val: np.ndarray, b_val: np.ndarray, target_val: np.ndarray
) -> Dict[str, Any]:
    """Mathematically verify Educational Autograd gradients against PyTorch Autograd.

    Computes: Loss = MSE(ReLU(X @ W + b), Target) in both engines.
    """
    # 1. Educational Autograd Pass
    x_edu = EducationalTensor(x_val)
    w_edu = EducationalTensor(w_val)
    b_edu = EducationalTensor(b_val)

    h_edu = (x_edu @ w_edu) + b_edu
    act_edu = h_edu.relu()
    loss_edu = act_edu.mse_loss(target_val)
    loss_edu.backward()

    # 2. PyTorch Autograd Pass
    x_torch = torch.tensor(x_val, dtype=torch.float64, requires_grad=False)
    w_torch = torch.tensor(w_val, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)
    target_torch = torch.tensor(target_val, dtype=torch.float64, requires_grad=False)

    h_torch = torch.matmul(x_torch, w_torch) + b_torch
    act_torch = torch.relu(h_torch)
    loss_torch = torch.mean((act_torch - target_torch) ** 2)
    loss_torch.backward()

    # 3. Calculate Maximum Absolute Gradient Difference
    w_diff = np.max(np.abs(w_edu.grad - w_torch.grad.detach().numpy()))
    b_diff = np.max(np.abs(b_edu.grad - b_torch.grad.detach().numpy()))
    loss_diff = abs(loss_edu.data.item() - loss_torch.item())

    return {
        "loss_educational": float(loss_edu.data),
        "loss_pytorch": float(loss_torch.item()),
        "loss_diff": float(loss_diff),
        "weight_grad_max_diff": float(w_diff),
        "bias_grad_max_diff": float(b_diff),
        "verified": bool(w_diff < 1e-6 and b_diff < 1e-6 and loss_diff < 1e-6),
    }
