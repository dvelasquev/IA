import torch
import torch.nn.functional as F
import math

class Net:
    def __init__(self, device='cuda'):
        self.layers = []
        self.device = device

    def add(self, layer):
        self.layers.append(layer)

    def train(self):
        for layer in self.layers:
            layer.training = True

    def eval(self):
        for layer in self.layers:
            layer.training = False

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
    def __init__(self, in_features, out_features, device='cuda'):
        limit = math.sqrt(2 / in_features)
        self.W = torch.randn(in_features, out_features, device=device) * limit
        self.b = torch.zeros(out_features, device=device)
        self.X, self.dW, self.db = None, None, None
        self.device = device

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.X.t() @ dZ
        self.db = dZ.sum(0)
        dX = dZ @ self.W.t()
        return dX

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
            self.mask = (torch.rand_like(X) > self.p).float() / (1 - self.p)
            return X * self.mask
        return X

    def backward(self, dX):
        return dX * self.mask if self.training else dX

    def update(self, lr):
        pass

class BatchNorm:
    def __init__(self, n_features, device='cuda'):
        self.gamma = torch.ones(n_features, device=device)
        self.beta = torch.zeros(n_features, device=device)
        self.running_mean = torch.zeros(n_features, device=device)
        self.running_var = torch.ones(n_features, device=device)
        self.eps = 1e-5
        self.momentum = 0.9
        self.training = True

    def forward(self, X):
        if self.training:
            self.mean = X.mean(0)
            self.var = X.var(0, unbiased=False)
            self.X_hat = (X - self.mean) / torch.sqrt(self.var + self.eps)
            self.running_mean = 0.9 * self.running_mean + 0.1 * self.mean
            self.running_var = 0.9 * self.running_var + 0.1 * self.var
        else:
            self.X_hat = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gamma * self.X_hat + self.beta

    def backward(self, dY):
        m = dY.shape[0]
        dX_hat = dY * self.gamma
        dvar = (dX_hat * (self.X - self.mean) * -0.5 * (self.var + self.eps)**-1.5).sum(0)
        dmean = (dX_hat * -1 / torch.sqrt(self.var + self.eps)).sum(0) + dvar * -2 * (self.X - self.mean).mean(0)
        dX = dX_hat / torch.sqrt(self.var + self.eps) + dvar * 2 * (self.X - self.mean) / m + dmean / m
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
        log_probs = F.log_softmax(Z, dim=1)
        return -log_probs[torch.arange(len(Y)), Y].mean()

    def backward(self, n_classes):
        Y_one_hot = F.one_hot(self.Y, num_classes=n_classes).float()
        dZ = (self.A - Y_one_hot) / len(self.Y)
        return dZ
