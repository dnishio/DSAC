import chainer
from chainer import functions as F
from chainer import links as L
import distribution


class FCHead(chainer.Chain):
    def __init__(self, n_hidden_size=(256, 256)):
        super().__init__()
        winit = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden_size[0], initialW=winit)
            self.l2 = L.Linear(None, n_hidden_size[1], initialW=winit)

    def __call__(self, x, is_absorb, a=None):
        x = F.concat((x, is_absorb), axis=-1)
        if a is not None:
            x = F.concat((x, a), axis=-1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return h

class CNNHead(chainer.Chain):
    def __init__(self, n_input_channels):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(n_input_channels, 16, 8, stride=4)
            self.l2 = L.Convolution2D(16, 32, 4, stride=2)
            self.l3 = L.Convolution2D(32, 32, 3, stride=1)
            self.l4 = L.Linear(None, 256)

    def __call__(self, x, is_absorb, a=None):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.concat((h, is_absorb), axis=-1)
        if a is not None:
            h = F.concat((h, a), axis=-1)
        h = F.relu(self.l4(h))
        return h

class GaussianPolicy(chainer.Chain):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        winit = chainer.initializers.GlorotUniform(1.)
        with self.init_scope():
            self.head = head
            self.out = L.Linear(None, action_size * 2, initialW=winit)

    def __call__(self, x, is_absorb):
        h = self.head(x, is_absorb)
        h = self.out(h)
        mean, log_scale = F.split_axis(h, 2, axis=1)
        log_scale = F.clip(log_scale, -20., 2.)
        var = F.exp(log_scale * 2)
        return distribution.SquashedGaussianDistribution(
            mean, var=var)

class SoftmaxPolicy(chainer.Chain):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        with self.init_scope():
            self.head = head
            self.out = L.Linear(None, action_size)

    def __call__(self, x, is_absorb):
        h = self.head(x, is_absorb)
        out = self.out(h)
        return distribution.SoftmaxDistribution(out)

class QSAFunction(chainer.Chain):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        with self.init_scope():
            self.head = head
            self.out = L.Linear(None, 1)

    def __call__(self, s, is_absorb, a):
        h = self.head(s, is_absorb, a)
        out = self.out(h)
        return out

class QSFunction(chainer.Chain):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        with self.init_scope():
            self.head = head
            self.out = L.Linear(None, action_size)

    def __call__(self, s, is_absorb):
        h = self.head(s, is_absorb)
        out = self.out(h)
        return out

# GAN for reward fnction
# class FCGanHead(chainer.Chain):
#     def __init__(self, n_hidden_size=256):
#         super().__init__()
#         winit = chainer.initializers.Orthogonal(scale=1.0)
#         with self.init_scope():
#             self.l1 = L.Linear(None, n_hidden_size, initialW=winit)
#             self.l2 = L.Linear(None, n_hidden_size, initialW=winit)
#
#     def __call__(self, x, is_absorb, a=None):
#         x = F.concat((x, is_absorb), axis=-1)
#         if a is not None:
#             x = F.concat((x, a), axis=-1)
#         h = F.tanh(self.l1(x))
#         h = F.tanh(self.l2(h))
#         return h
#
# class CNNGanHead(chainer.Chain):
#     def __init__(self, n_input_channels):
#         super().__init__()
#         with self.init_scope():
#             self.l1 = L.Convolution2D(n_input_channels, 16, 8, stride=4)
#             self.l2 = L.Convolution2D(16, 32, 4, stride=2)
#             self.l3 = L.Convolution2D(32, 32, 3, stride=1)
#             self.l4 = L.Linear(None, 256)
#
#     def __call__(self, x, is_absorb, a=None):
#         h = F.relu(self.l1(x))
#         h = F.relu(self.l2(h))
#         h = F.relu(self.l3(h))
#         h = F.concat((h, is_absorb), axis=-1)
#         if a is not None:
#             h = F.concat((h, a), axis=-1)
#         h = F.relu(self.l4(h))
#         return h

class FCRewardFunction(chainer.Chain):
    def __init__(self, action_size, n_hidden_size=100):
        super().__init__()
        self.action_size = action_size
        # winit = chainer.initializers.Orthogonal(scale=1.0)
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden_size)
            self.l2 = L.Linear(None, n_hidden_size)
            self.out = L.Linear(None, 1)

    def __call__(self, x, is_absorb, a=None):
        x = F.concat((x, is_absorb), axis=-1)
        if a is not None:
            x = F.concat((x, a), axis=-1)
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        out = self.out(h)
        return out

    def forward(self, x):
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        out = self.out(h)
        return out
