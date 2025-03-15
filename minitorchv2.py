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


class Linear:
    def __init__(self, in_features, out_features, device="cuda"):
        limit = math.sqrt(2 / in_features)
        self.W = torch.randn(in_features, out_features, device=device) * limit
        self.b = torch.zeros(out_features, device=device)
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.X.t() @ dZ
        self.db = dZ.sum(dim=0)
        return dZ @ self.W.t()

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask

    def backward(self, dA):
        return dA * self.mask

    def update(self, lr):
        pass


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = (torch.rand_like(X) > self.p).float() / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, dX):
        if self.training:
            return dX * self.mask
        return dX

    def update(self, lr):
        pass


class BatchNorm:
    def __init__(self, n_features, device="cuda"):
        self.gamma = torch.ones(n_features, device=device)
        self.beta = torch.zeros(n_features, device=device)
        self.running_mean = torch.zeros(n_features, device=device)
        self.running_var = torch.ones(n_features, device=device)
        self.eps = 1e-5
        self.momentum = 0.9

    def forward(self, X):
        if self.training:
            self.mean = X.mean(0)
            self.var = X.var(dim=0)
            self.X_hat = (X - self.mean) / torch.sqrt(self.var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * self.var
        else:
            self.X_hat = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gamma * self.X_hat + self.beta

    def backward(self, dY):
        batch_size = dX.shape[0]
        self.dgamma = (dX * self.X_hat).sum(dim=0)
        self.dbeta = dX.sum(dim=0)

        dX_hat = dX * self.gamma
        dvar = (-0.5 * (dX_hat * (self.X - self.mean)).sum(dim=0)) * ((self.var + self.eps)**(-1.5))
        dmean = (-dX_hat / torch.sqrt(self.var + self.eps)).sum(dim=0) + dvar * (-2 / batch_size * (self.X - self.mean).sum(dim=0))

        dX = dX_hat / torch.sqrt(self.var + self.eps) + dvar * 2 * (self.X - self.mean) / batch_size + dmean / batch_size

        return dX

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


class CrossEntropyFromLogits:
    def forward(self, Z, Y):
        self.Y = Y
        self.A = torch.softmax(Z, dim=1)
        log_softmax_Z = torch.log_softmax(Z, dim=1)
        loss = -log_softmax_Z[torch.arange(len(Y)), Y].mean()
        return loss

    def backward(self, n_classes):
        Y_one_hot = torch.nn.functional.one_hot(self.Y, num_classes=n_classes).float()
        dZ = (self.A - Y_one_hot) / self.Y.shape[0]
        return dZ
