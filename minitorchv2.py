import torch
import torch.nn.functional as F
import math


class Net:
    def __init__(self, device="cuda"):
        self.layers = []
        self.device = device

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        X = X.to(self.device)
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False


class Linear:
    def __init__(self, in_features, out_features, device="cuda"):
        self.W = torch.randn(in_features, out_features, device=device) * math.sqrt(2 / in_features)
        self.b = torch.zeros(out_features, device=device)
        self.device = device

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.X.t() @ dZ
        self.db = dZ.sum(0)
        return dZ @ self.W.t()

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def forward(self, Z):
        self.Z = Z
        return torch.relu(Z)

    def backward(self, dA):
        return dA * (self.Z > 0).float()

    def update(self, lr):
        pass


class Dropout:
    def __init__(self, p=0.5, device="cuda"):
        self.p = p
        self.training = True
        self.device = device

    def forward(self, X):
        if self.training:
            self.mask = (torch.rand_like(X) > self.p).float()
            return X * self.mask / (1 - self.p)
        return X

    def backward(self, dA):
        if self.training:
            return dA * self.mask / (1 - self.p)
        return dA

    def update(self, lr):
        pass


class BatchNorm:
    def __init__(self, num_features, device="cuda", eps=1e-5):
        self.gamma = torch.ones(num_features, device=device)
        self.beta = torch.zeros(num_features, device=device)
        self.eps = eps
        self.device = device

    def forward(self, X):
        self.mu = X.mean(0)
        self.var = X.var(0)
        self.X_hat = (X - self.mu) / torch.sqrt(self.var + self.eps)
        return self.gamma * self.X_hat + self.beta

    def backward(self, dY):
        N = dY.shape[0]
        std_inv = 1. / torch.sqrt(self.var + self.eps)

        dX_hat = dY * self.gamma
        dvar = (dX_hat * (self.X_hat * -0.5) * std_inv**3).sum(0)
        dmu = (dX_hat * -std_inv).sum(0) + dvar * (-2.0 * self.X_hat.mean(0))

        dX = dX_hat * std_inv + dvar * 2 * self.X_hat / N + dmu / N
        self.dgamma = (dY * self.X_hat).sum(0)
        self.dbeta = dY.sum(0)
        return dX

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


class CrossEntropyFromLogits:
    def forward(self, Z, Y):
        self.Y = Y
        self.A = F.softmax(Z, dim=1)
        log_softmax_Z = F.log_softmax(Z, dim=1)
        log_probs = log_softmax_Z[torch.arange(Z.size(0)), Y]
        return -log_probs.mean()

    def backward(self, n_classes):
        Y_one_hot = F.one_hot(self.Y, num_classes=n_classes).float()
        return (self.A - Y_one_hot) / self.Y.size(0)
