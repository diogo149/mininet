import numpy as np


class Variable(object):

    def __init__(self, shape):
        self.shape = shape
        self.value = None
        self.grad = None


class BaseLayer(object):

    def fprop(self):
        """
        replaces out_var with correct value
        """
        raise NotImplementedError

    def bprop(self):
        """
        fills gradients of all input vars, assuming gradient of out_var
        is provided
        """
        raise NotImplementedError


class BiasLayer(BaseLayer):

    def __init__(self, in_var, bias):
        self.in_var = in_var
        self.bias = bias
        out_shape = self.in_var.shape
        self.out_var = Variable(out_shape)

    def fprop(self):
        # add batch axis to bias
        self.out_var.value = self.in_var.value + self.bias.value[np.newaxis]

    def bprop(self):
        # dz/dx = 1
        self.in_var.grad = self.out_var.grad
        # sum across minibatch axis
        self.bias.grad = self.out_var.grad.sum(axis=0)


class LinearLayer(BaseLayer):

    def __init__(self, in_var, weight):
        self.in_var = in_var
        self.weight = weight
        out_shape = (self.in_var.shape[0], self.weight.shape[1])
        self.out_var = Variable(out_shape)

    def fprop(self):
        self.out_var.value = np.dot(self.in_var.value, self.weight.value)

    def bprop(self):
        self.in_var.grad = np.dot(self.out_var.grad, self.weight.value.T)
        self.weight.grad = np.dot(self.in_var.value.T, self.out_var.grad)


class SigmoidLayer(BaseLayer):

    def __init__(self, in_var):
        self.in_var = in_var
        out_shape = self.in_var.shape
        self.out_var = Variable(out_shape)

    def fprop(self):
        self.out_var.value = 1. / (1 + np.exp(-self.in_var.value))

    def bprop(self):
        g = self.out_var.value
        self.in_var.grad = self.out_var.grad * g * (1 - g)


class MeanSquaredCostLayer(BaseLayer):

    def __init__(self, in_var, target):
        # NOTE: doesn't calc gradient of target
        self.in_var = in_var
        self.target = target
        out_shape = ()
        self.out_var = Variable(out_shape)

    def fprop(self):
        self._err = self.in_var.value - self.target.value
        self.out_var.value = np.mean(self._err ** 2)

    def bprop(self):
        n = self.in_var.value.size
        self.in_var.grad = 2 * self._err / n


if __name__ == "__main__":
    # example autoencoder

    layers = []
    params = []

    def fc(in_var, num_units):
        weight = Variable((in_var.shape[1], num_units))
        bias = Variable((num_units,))
        params.append(weight)
        params.append(bias)
        linear = LinearLayer(in_var, weight)
        layers.append(linear)
        bias = BiasLayer(linear.out_var, bias)
        layers.append(bias)
        return bias.out_var

    def sigmoid(in_var):
        l = SigmoidLayer(in_var)
        layers.append(l)
        return l.out_var

    def mse(in_var, target):
        l = MeanSquaredCostLayer(in_var, target)
        layers.append(l)
        return l.out_var

    def fprop():
        for l in layers:
            l.fprop()

    def bprop():
        for l in reversed(layers):
            l.bprop()

    def sgd():
        for param in params:
            param.value += - learning_rate * param.grad

    batch_size = 8
    num_inputs = 8
    learning_rate = 0.1
    x = Variable((batch_size, num_inputs))
    h = sigmoid(fc(x, 3))
    y_hat = sigmoid(fc(h, 8))
    cost = mse(y_hat, x)

    for param in params:
        param.value = 0.1 * np.random.randn(*param.shape)

    minibatch = np.eye(8)
    for i in range(10000000):
        x.value = minibatch
        fprop()
        if ((i + 1) % 1000) == 0:
            print cost.value
        bprop()
        sgd()
