import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T

from iRBM.models.orbm import oRBM
from iRBM.training import tasks

from iRBM.misc.utils import logsumexp
from iRBM.misc.regularization import L1Regularization, L2Regularization


class iRBM(oRBM):
    """Infinite Restricted Boltzmann Machine (iRBM)  """
    def __init__(self,
                 input_size,
                 hidden_size,
                 beta=1,
                 max_hidden_size=20000,
                 *args, **kwargs):
        oRBM.__init__(self, input_size, hidden_size, beta, *args, **kwargs)

        self.W.set_value(np.zeros_like(self.W.get_value()))
        self.max_hidden_size = max_hidden_size

    def free_energy_zmask(self, v, zmask):
        """ Marginalization over hidden units"""
        free_energy = -T.dot(v, self.c) - logsumexp(self.log_z_given_v(v)*zmask, axis=1)  # Sum over z'
        return free_energy

    def pdf_z_given_v(self, v, method="infinite"):
        return oRBM.pdf_z_given_v(self, v, method=method)

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


class GrowiRBM(tasks.Task):
    def __init__(self, model, shrinkable=False, nb_neurons_to_add=1):
        super(GrowiRBM, self).__init__()
        self.model = model
        self.shrinkable = shrinkable
        self.nb_neurons_to_add = nb_neurons_to_add
        self.maxZ = theano.shared(np.array(0, dtype="int64"))
        self.grad_W_new_neurons = theano.shared(np.zeros((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX))

        zmask_start = model.sample_zmask_given_v(model.CD.chain_start)
        zmask_end = model.sample_zmask_given_v(model.CD.chain_end)
        z_start = T.sum(zmask_start, axis=1)
        z_end = T.sum(zmask_end, axis=1)
        max_Zs = T.maximum(z_start, z_end)
        maxZ = max_Zs.max()

        W_bak = model.W
        b_bak = model.b
        model.W = T.join(0, model.W, T.zeros((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX))
        model.b = T.join(0, model.b, T.zeros(nb_neurons_to_add, dtype=theano.config.floatX))
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
            #raise NotImplementedError

            #unused_units = np.bitwise_and(np.all(model.W.get_value() == 0., axis=1),
            #                              model.b.get_value() == 0.)
            #from ipdb import set_trace as dbg
            #dbg()
            #nb_neurons_to_remove = np.diff(np.cumsum(unused_units[::-1]))

            decrease_needed = np.all(model.W.get_value()[-1] == 0.) and model.b.get_value()[-1] == 0
            if decrease_needed:
                nb_neurons_to_del = 1
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
