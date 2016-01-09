import copy

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

from iRBM.models.model import Model
from iRBM.misc.contrastive_divergence import ContrastiveDivergence


class RBM(Model):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input_size,
                       hidden_size,
                       CD=ContrastiveDivergence(),
                       CDk=1,
                       *args, **kwargs):

        Model.__init__(self, *args, **kwargs)

        self.CD = CD
        self.CDk = CDk

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = theano.shared(value=np.zeros((self.hidden_size, self.input_size), dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=np.zeros(self.hidden_size,                    dtype=theano.config.floatX), name='b')
        self.c = theano.shared(value=np.zeros(self.input_size,                     dtype=theano.config.floatX), name='c')

        self.parameters = [self.W, self.b, self.c]
        self.setup()

    def setup(self):
        W = 1e-2*self.np_rng.randn(self.hidden_size, self.input_size).astype(theano.config.floatX)
        self.W.set_value(W)

    def E(self, v, h):
        energy = -T.dot(v, self.c)
        energy += (-T.dot(T.dot(h, self.W), v.T).T - T.dot(h, self.b)).T
        return energy

    def free_energy(self, v):
        return -T.dot(v, self.c) - T.sum(T.nnet.softplus(T.dot(v, self.W.T) + self.b), axis=1)

    def marginalize_over_v(self, h):
        return -T.dot(h, self.b) - T.sum(T.nnet.softplus(T.dot(h, self.W) + self.c), axis=1)

    def sample_h_given_v(self, v, return_probs=False):
        pre_sigmoid_activation = T.dot(v, self.W.T) + self.b
        h_mean = T.nnet.sigmoid(pre_sigmoid_activation)

        if return_probs:
            return h_mean

        h_sample = self.theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean, dtype=theano.config.floatX)
        return h_sample

    def sample_v_given_h(self, h, return_probs=False):
        pre_sigmoid_activation = T.dot(h, self.W) + self.c
        x_mean = T.nnet.sigmoid(pre_sigmoid_activation)

        if return_probs:
            return x_mean

        x_sample = self.theano_rng.binomial(size=x_mean.shape, n=1, p=x_mean, dtype=theano.config.floatX)
        return x_sample

    def gibbs_step(self, v0):
        h0 = self.sample_h_given_v(v0)
        v1 = self.sample_v_given_h(h0)
        return v1

    @property
    def regularization(self):
        return self.regularize(self.W)

    def get_updates(self, v):
        # Contrastive divergence
        chain_end, updates_CD = self.CD(self, chain_start=v, cdk=self.CDk)

        # [Expected] negative log-likelihood
        cost = T.mean(self.free_energy(v), axis=0) - T.mean(self.free_energy(chain_end), axis=0)

        #Regularization
        cost += self.regularization

        # Gradients (use automatic differentiation)
        # We must not compute the gradient through the gibbs sampling, i.e. use consider_constant
        gparams = T.grad(cost, self.parameters, consider_constant=[chain_end])
        gradients = dict(zip(self.parameters, gparams))

        # Get learning rates for all params given their gradient.
        lr, updates_lr = self.learning_rate(gradients)

        updates = OrderedDict()
        updates.update(updates_CD)  # Add updates from CD
        updates.update(updates_lr)  # Add updates from learning_rate

        # Updates parameters
        for param, gparam in gradients.items():
            updates[param] = param - lr[param] * gradients[param]

        return updates
        #return updates, (v, chain_end)  # TEMP: to test L1 regul

    def __getstate__(self):
        state = {}
        state.update(Model.__getstate__(self))
        state['RBM_version'] = 1

        # Save parameters (e.g. W, b, c)
        state["parameters"] = {}
        for param in self.parameters:
            state["parameters"][param.name] = param.get_value()

        # Hyper parameters
        state['input_size'] = self.input_size
        state['hidden_size'] = self.hidden_size
        state['CD'] = self.CD
        state['CDk'] = self.CDk

        return state

    def __setstate__(self, state):
        Model.__setstate__(self, state)

        # Save parameters (e.g. W, b, c)
        self.parameters = []
        for name, value in state['parameters'].items():
            param = theano.shared(value, name=name)
            setattr(self, name, param)
            self.parameters.append(param)

        # Hyper parameters
        self.input_size = state['input_size']
        self.hidden_size = state['hidden_size']
        self.CDk = state['CDk']
        self.CD = state['CD']

    def get_base_rate(self, base_rate_type="uniform"):
        """ base_rate_type = {'uniform', 'c', 'b'} """
        base_rate = copy.deepcopy(self)
        base_rate.W = T.zeros_like(self.W)

        annealable_params = [self.W]

        if base_rate_type == "uniform":
            base_rate.b = T.zeros_like(self.b)
            base_rate.c = T.zeros_like(self.c)
            annealable_params.append(self.b)
            annealable_params.append(self.c)

            def compute_lnZ(self):
                # Since all parameters are 0, there are 2^hidden_size and 2^input_size
                # different neuron's states having the same energy (i.e. E=1)
                return self.input_size * T.log(2) + self.hidden_size * T.log(2)

        elif base_rate_type == "c":
            base_rate.b = T.zeros_like(self.b)
            annealable_params.append(self.b)

            def compute_lnZ(self):
                # Since all parameters are 0 except visible biases, there are 2^hidden_size
                # different hidden neuron's states having the same marginalization over h.
                lnZ = -self.marginalize_over_v(h=T.zeros((1, self.hidden_size)))
                lnZ += self.hidden_size * T.log(2)
                return lnZ[0]

        elif base_rate_type == "b":
            base_rate.c = T.zeros_like(self.c)
            annealable_params.append(self.c)

            def compute_lnZ(self):
                # Since all parameters are 0 except hidden biases, there are 2^input_size
                # different visible neuron's states having the same free energy (i.e. marginalization over v).
                lnZ = -self.free_energy(v=T.zeros((1, self.input_size)))
                lnZ += self.input_size * T.log(2)
                return lnZ[0]

        import types
        base_rate.compute_lnZ = types.MethodType(compute_lnZ, base_rate)

        return base_rate, annealable_params
