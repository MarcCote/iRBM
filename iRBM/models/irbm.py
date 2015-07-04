import numpy as np

import theano
import theano.tensor as T

from iRBM.models.orbm import oRBM
from iRBM.training import tasks

from iRBM.misc.utils import logsumexp


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

    @property
    def regularization(self):
        return self.regularize(self.W) + self.regularize(self.b)

    def __getstate__(self):
        state = {}
        state.update(oRBM.__getstate__(self))
        state['iRBM_version'] = 2
        state['max_hidden_size'] = self.max_hidden_size

        return state

    def __setstate__(self, state):
        oRBM.__setstate__(self, state)

        if state['iRBM_version'] >= 2:
            self.max_hidden_size = state['max_hidden_size']


class GrowiRBM(tasks.Task):
    def __init__(self, model):
        super(GrowiRBM, self).__init__()
        self.model = model
        self.maxZ = theano.shared(np.array(0, dtype="int64"))

        zmask_start = model.sample_zmask_given_v(model.CD.chain_start)
        zmask_end = model.sample_zmask_given_v(model.CD.chain_end)

        # Will be part of the updates passed to the Theano function `learn` of the trainer.
        self.updates[self.maxZ] = T.cast(T.maximum(T.sum(zmask_start, axis=1), T.sum(zmask_end, axis=1)).max(), "int64")

    def post_update(self, no_epoch, no_update):
        model = self.model
        increase_needed = model.hidden_size < model.max_hidden_size and self.maxZ.get_value() == model.hidden_size

        if increase_needed:
            nb_neurons_to_add = 1  # min(model.hidden_size*2-model.hidden_size, 100)

            model.hidden_size += nb_neurons_to_add
            model.W.set_value(np.r_[model.W.get_value(), np.zeros((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX)])
            model.b.set_value(np.r_[model.b.get_value(), np.zeros(nb_neurons_to_add, dtype=theano.config.floatX)])

            # Also increase update rule params, if needed.
            for lr_param in model.learning_rate.parameters:
                if model.W.name in lr_param.name:
                    lr_param.set_value(np.r_[lr_param.get_value(), np.zeros((nb_neurons_to_add, model.input_size), dtype=theano.config.floatX)])
                elif model.b.name in lr_param.name:
                    lr_param.set_value(np.r_[lr_param.get_value(), np.zeros(nb_neurons_to_add, dtype=theano.config.floatX)])

    def post_epoch(self, no_epoch, no_update):
        print("Hidden size: {}".format(self.model.hidden_size))
