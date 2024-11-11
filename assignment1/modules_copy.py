################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        self.params = {
            'weight': np.random.randn(out_features, in_features) * np.sqrt(2. / in_features),
            'bias': np.zeros(out_features)
        }
        self.grads = {'weight': np.zeros_like(self.params['weight']), 'bias': np.zeros_like(self.params['bias'])}

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.params['weight'].T) + self.params['bias']
        return out

    def backward(self, dout):
        self.grads['weight'] = np.dot(dout.T, self.x)
        self.grads['bias'] = np.sum(dout, axis=0)
        dx = np.dot(dout, self.params['weight'])
        return dx

    def clear_cache(self):
        self.x = None


class ELUModule(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        out = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
        return out

    def backward(self, dout):
        dx = dout * np.where(self.x >= 0, 1, self.alpha * np.exp(self.x))
        return dx

    def clear_cache(self):
        self.x = None


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        out = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
        return out

    def backward(self, dout):
        dx = dout * np.where(self.x >= 0, 1, self.alpha * np.exp(self.x))
        return dx

    def clear_cache(self):
        self.x = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_shifted)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        dx = self.out * (dout - np.sum(self.out * dout, axis=1, keepdims=True))
        return dx

    def clear_cache(self):
        self.out = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        m = y.shape[0]
        p = np.exp(x - np.max(x, axis=1, keepdims=True))
        p /= np.sum(p, axis=1, keepdims=True)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, x, y):
        m = y.shape[0]
        grad = np.exp(x - np.max(x, axis=1, keepdims=True))
        grad /= np.sum(grad, axis=1, keepdims=True)
        grad[range(m), y] -= 1
        dx = grad / m
        return dx