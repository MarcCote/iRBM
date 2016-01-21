import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T

from iRBM.models.rbm import RBM
from iRBM.models.orbm import oRBM
from iRBM.training import tasks

from iRBM.misc.utils import logsumexp
from iRBM.misc.regularization import L1Regularization, L2Regularization


class iRBM(oRBM):
    """Infinite Restricted Boltzmann Machine (iRBM)  """
    def __init__(self,
                 input_size,
                 hidden_size=1,
                 beta=1,
                 max_hidden_size=20000,
                 *args, **kwargs):
        oRBM.__init__(self, input_size, hidden_size, beta, *args, **kwargs)

        self.W.set_value(np.zeros_like(self.W.get_value()))
        self.max_hidden_size = max_hidden_size

    def F(self, v, z):
        energy = -T.dot(v, self.c)
        energy += -T.sum(T.nnet.softplus(T.dot(v, self.W[:z, :].T) + self.b[:z]), axis=1)

        # Add penality term
        if self.penalty == "softplus_bi":
            energy += T.sum(self.beta*T.log(1+T.exp(self.b[:z])))
        elif self.penalty == "softplus0":
            energy += T.sum(self.beta*T.log(1+T.exp(0)))
        else:
            raise NameError("Invalid penalty term")

        return energy

    def free_energy(self, v):
        """ Marginalization over hidden units"""
        free_energy = -T.dot(v, self.c) - logsumexp(self.log_z_given_v(v), axis=1)  # Sum over z'
        return free_energy

    def free_energy_zmask(self, v, zmask):
        """ Marginalization over hidden units"""
        free_energy = -T.dot(v, self.c) - logsumexp(self.log_z_given_v(v)*zmask, axis=1)  # Sum over z'
        return free_energy

    def log_z_given_v(self, v):
        log_z_given_v = oRBM.log_z_given_v(self, v)

        geometric_ratio = T.exp((1.-self.beta) * T.nnet.softplus(0.)).eval()
        #log_shifted_geometric_convergence = np.float32(np.log(geometric_ratio / (1. - geometric_ratio)))
        log_geometric_convergence = np.float32(np.log(1 / (1. - geometric_ratio)))

        # We add the remaining of the geometric series in the last bucket.
        log_z_given_v = T.set_subtensor(log_z_given_v[:, -1], log_z_given_v[:, -1] + log_geometric_convergence)
        return log_z_given_v

    def pdf_z_given_v(self, v):
        log_z_given_v = self.log_z_given_v(v)
        log_sum_z_given_v = logsumexp(log_z_given_v, axis=1)
        prob_z_given_v = T.exp(log_z_given_v - log_sum_z_given_v[:, None])
        return prob_z_given_v

    def get_updates(self, v):
        # Contrastive divergence
        chain_end, updates_CD = self.CD(self, chain_start=v, cdk=self.CDk)

        # [Expected] negative log-likelihood
        cost = T.mean(self.free_energy(v), axis=0) - T.mean(self.free_energy(chain_end), axis=0)

        # L2 Regularization
        if isinstance(self.regularize, L2Regularization):
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

        if isinstance(self.regularize, L1Regularization):
            updates[self.b] = T.sgn(updates[self.b]) * T.maximum(abs(updates[self.b]) - lr[self.b]*self.regularize.decay, 0)
            updates[self.W] = T.sgn(updates[self.W]) * T.maximum(abs(updates[self.W]) - lr[self.W]*self.regularize.decay, 0)

        return updates

    def __getstate__(self):
        state = {}
        state.update(oRBM.__getstate__(self))
        state['iRBM_version'] = 3
        state['max_hidden_size'] = self.max_hidden_size

        return state

    def __setstate__(self, state):
        oRBM.__setstate__(self, state)

        if state['iRBM_version'] >= 2:
            self.max_hidden_size = state['max_hidden_size']

    def get_base_rate(self, base_rate_type="uniform"):
        base_rate, annealable_params = RBM.get_base_rate(self, base_rate_type)
        #annealable_params.append(self.beta)  # Seems to work better without annealing self.beta (see unit tests)

        if base_rate_type == "uniform":
            def compute_lnZ(self):
                # Since biases and weights are all 0, there are $2^input_size$ different
                #  visible neuron's states having the following energy
                #  $\sum_{z=1}^H \sum_{h \in \{0,1\}^z} \exp(-\beta z \ln(2))$
                r = T.exp((1-self.beta) * T.log(2))  # Ratio of a geometric serie
                lnZ = T.log(r / (1-r))  # Convergence of the geometric serie
                return (self.input_size * T.log(2) +  # ln(2^input_size)
                        lnZ)  # $ln( \sum_{z=1}^H \sum_{h \in \{0,1\}^z} \exp(-\beta z \ln(2)) )$

        elif base_rate_type == "c":
            def compute_lnZ(self):
                # Since the hidden biases (but not the visible ones) and the weights are all 0
                r = T.exp((1-self.beta) * T.log(2))  # Ratio of a geometric serie
                lnZ = T.log(r / (1-r))  # Convergence of the geometric serie
                return (lnZ +  # $ln( \sum_{z=1}^H \sum_{h \in \{0,1\}^z} \exp(-\beta z \ln(2)) )$
                        T.sum(T.nnet.softplus(self.c)))

        elif base_rate_type == "b":
            raise NotImplementedError()

        import types
        base_rate.compute_lnZ = types.MethodType(compute_lnZ, base_rate)

        return base_rate, annealable_params


class GrowiRBM(tasks.Task):
    def __init__(self, model, shrinkable=False, nb_neurons_to_add=1, random_init=False):
        super(GrowiRBM, self).__init__()
        self.model = model
        self.shrinkable = shrinkable
        self.random_init = random_init
        self.nb_neurons_to_add = nb_neurons_to_add
        self.maxZ = theano.shared(np.array(0, dtype="int64"))
        self.grad_W_new_neurons = theano.shared(np.zeros((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX))

        zmask_start = model.sample_zmask_given_v(model.CD.chain_start)
        zmask_end = model.sample_zmask_given_v(model.CD.chain_end)
        z_start = T.sum(zmask_start, axis=1)
        z_end = T.sum(zmask_end, axis=1)
        max_Zs = T.maximum(z_start, z_end)
        maxZ = max_Zs.max()

        W_init = T.zeros((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX)
        b_init = T.zeros(nb_neurons_to_add, dtype=theano.config.floatX)

        if self.random_init:
            from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
            trng = RandomStreams(42)
            W_init = 1e-2 * trng.normal((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX)

        W_bak = model.W
        b_bak = model.b
        model.W = T.join(0, model.W, W_init)
        model.b = T.join(0, model.b, b_init)
        cost = model.free_energy(model.CD.chain_start) - model.free_energy(model.CD.chain_end)
        grad_W_new_neurons = theano.grad(T.mean(cost), model.W)[-nb_neurons_to_add:]
        model.W = W_bak
        model.b = b_bak

        # Will be part of the updates passed to the Theano function `learn` of the trainer.
        # Notes: all updates are done simultanously, i.e. params haven't been updated yet.
        self.updates[self.maxZ] = T.cast(maxZ, "int64")
        self.updates[self.grad_W_new_neurons] = grad_W_new_neurons

        # For debugging
        # self.z_start = theano.shared(np.zeros(64, dtype="int64"))
        # self.z_end = theano.shared(np.zeros(64, dtype="int64"))
        # self.updates[self.z_start] = T.cast(z_start, "int64")
        # self.updates[self.z_end] = T.cast(z_end, "int64")

    def post_update(self, no_epoch, no_update):
        model = self.model
        increase_needed = model.hidden_size < model.max_hidden_size and self.maxZ.get_value() == model.hidden_size

        if increase_needed:
            model.hidden_size += self.nb_neurons_to_add

            grad_W_new_neurons = self.grad_W_new_neurons.get_value()

            # Also increase update rule params, if needed.
            for lr_param in model.learning_rate.parameters:
                if model.W.name in lr_param.name:
                    # Assume this is ADAGRAD, so we store the gradient squared
                    lr_param.set_value(np.r_[lr_param.get_value(), grad_W_new_neurons**2])
                elif model.b.name in lr_param.name:
                    lr_param.set_value(np.r_[lr_param.get_value(), np.zeros(self.nb_neurons_to_add, dtype=theano.config.floatX)])

            # Compute the learning rate associated to parameter W of the new hidden units.
            lr_W = model.learning_rate.get_lr(model.W)
            W_new_hidden_units = -lr_W[-self.nb_neurons_to_add:]*grad_W_new_neurons

            # Apply regularization
            if isinstance(model.regularize, L1Regularization):
                W_new_hidden_units = np.sign(W_new_hidden_units) * np.maximum(abs(W_new_hidden_units) - lr_W[-self.nb_neurons_to_add:]*model.regularize.decay, 0)
            elif isinstance(model.regularize, L2Regularization):
                W_new_hidden_units = model.regularize(W_new_hidden_units)

            # Append params associated to the new hidden units.
            model.W.set_value(np.r_[model.W.get_value(), W_new_hidden_units])
            model.b.set_value(np.r_[model.b.get_value(), np.zeros(self.nb_neurons_to_add, dtype=theano.config.floatX)])

        if self.shrinkable:
            # unused_units = np.bitwise_and(np.all(model.W.get_value() == 0., axis=1),
            #                               model.b.get_value() == 0.)

            # if unused_units[-1]:
            #     nb_neurons_to_del = np.where(np.diff(np.cumsum(unused_units[::-1])) == 0)[0][0] + 1

            decrease_needed = np.all(model.W.get_value()[-1] == 0.) and model.b.get_value()[-1] == 0
            if decrease_needed:
                nb_neurons_to_del = 1  # Shrink only one at the time.
                model.hidden_size -= nb_neurons_to_del
                model.W.set_value(model.W.get_value()[:-nb_neurons_to_del])
                model.b.set_value(model.b.get_value()[:-nb_neurons_to_del])

                # Also decrease update rule params, if needed.
                for lr_param in model.learning_rate.parameters:
                    if model.W.name in lr_param.name:
                        lr_param.set_value(lr_param.get_value()[:-nb_neurons_to_del])
                    elif model.b.name in lr_param.name:
                        lr_param.set_value(lr_param.get_value()[:-nb_neurons_to_del])

    def post_epoch(self, no_epoch, no_update):
        print("Hidden size: {}".format(self.model.hidden_size))
