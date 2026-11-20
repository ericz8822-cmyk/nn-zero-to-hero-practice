import torch

g = torch.Generator().manual_seed(2147483647)

class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias == None else [self.bias]) 

class BatchNorm1d:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True # using the running mean/std or not
        # parameters (corresponding to bngain and bnbias)
        # gamma and beta are the variables used in pytorch API
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (running mean and standard deviation bnmean and bnstd) 
        self.running_mean = torch.zeros(dim)
        self.running_std = torch.ones(dim)

    def __call__(self,x):
        # calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdims=True)
            xstd = x.std(dim, keepdims=True)
        else:
            xmean = self.running_mean
            xstd = self.running_std
        x_gaussian = (x - xmean) / torch.sqrt(xstd**2 + self.eps)
        self.out = self.gamma * x_gaussian + self.beta
        # updating the buffers
        if self.training:
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*xmean
            self.running_std = (1-self.momentum)*self.running_std + self.momentum*xstd
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:

    def __init__(self, num_embedding, embedding_dim):
        self.weight = torch.randn((num_embedding, embedding_dim))
 
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C, = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # get all the parameters of all layers in self.layer, turn them into a list
        return [p for layer in self.layers for p in layer.parameters()]
