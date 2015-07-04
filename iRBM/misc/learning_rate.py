import numpy as np

import theano
import theano.tensor as T
from theano import config

from collections import OrderedDict, defaultdict


def sharedX(value, name=None, borrow=False):
    """ Transform value into a shared variable of type floatX """
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


class CustomDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        return defaultdict.__getitem__(self, str(key))

    def __setitem__(self, key, val):
        defaultdict.__setitem__(self, str(key), val)


class CustomDict(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, str(key))

    def __setitem__(self, key, val):
        dict.__setitem__(self, str(key), val)


class LearningRate:
    def __init__(self, lr):
        self.base_lr = lr
        self.lr = CustomDefaultDict(lambda: theano.shared(np.array(lr, dtype=config.floatX)))

    def set_individual_lr(self, param, lr):
        self.lr[param].set_value(lr)

    def __call__(self, gradients):
        raise NameError('Should be implemented by inheriting class!')

    def __getstate__(self):
        # Convert defaultdict into a dict
        self.__dict__.update({"lr": CustomDict(self.lr)})
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

        if type(self.lr) is not CustomDict:
            self.lr = CustomDict()
            for k, v in state['lr'].items():
                self.lr[k] = v

        # Make sure each learning rate have the right dtype
        self.lr = CustomDict({k: theano.shared(v.get_value().astype(config.floatX), name='lr_' + k) for k, v in self.lr.items()})


class ConstantLearningRate(LearningRate):
    def __init__(self, lr):
        LearningRate.__init__(self, lr)

    def __call__(self, gradients):
        return self.lr, OrderedDict()


class ADAGRAD(LearningRate):
    def __init__(self, lr, eps=1e-6):
        """
        Implements the ADAGRAD learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        eps: float
            eps needed to avoid division by zero.

        Reference
        ---------
        Duchi, J., Hazan, E., & Singer, Y. (2010).
        Adaptive subgradient methods for online learning and stochastic optimization.
        Journal of Machine Learning
        """
        LearningRate.__init__(self, lr)

        self.epsilon = eps
        self.parameters = []

    def __call__(self, grads):
        updates = OrderedDict()
        learning_rates = OrderedDict()

        params_names = map(lambda p: p.name, self.parameters)
        for param in grads.keys():
            # sum_squared_grad := \sum g_t^2
            sum_squared_grad = sharedX(param.get_value() * 0.)

            if param.name is not None:
                sum_squared_grad.name = 'sum_squared_grad_' + param.name

            # Check if param is already there before adding
            if sum_squared_grad.name not in params_names:
                self.parameters.append(sum_squared_grad)
            else:
                sum_squared_grad = self.parameters[params_names.index(sum_squared_grad.name)]

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(grads[param])

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.epsilon)

            # Apply update
            updates[sum_squared_grad] = new_sum_squared_grad
            learning_rates[param] = self.base_lr / root_sum_squared

        return learning_rates, updates
