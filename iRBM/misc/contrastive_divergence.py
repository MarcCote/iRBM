import theano
import numpy as np

from collections import OrderedDict
import theano.tensor as T


class ContrastiveDivergence():
    def __init__(self):
        self.is_PCD = False
        self.chain_start = None
        self.chain_end = None

    def __call__(self, model, chain_start, cdk=1):
        """
        Parameters
        ----------
        model : `mlpython.learners.RBM` instance
            rbm-like model implemeting `gibbs_step` method
        chain_start : Theano variable
            argument to past to `gibbs_step` method
        cdk : int
            number of Gibbs step to do
        """
        if cdk == 1:
            chain_end, updates = model.gibbs_step(chain_start), OrderedDict()
        else:
            chain, updates = theano.scan(model.gibbs_step,
                                         outputs_info=chain_start,
                                         n_steps=cdk)

            chain_end = chain[-1]

        # Keep reference of chain_start and chain_end for later use.
        self.chain_start = chain_start
        self.chain_end = chain_end

        return chain_end, updates


class PersistentCD(ContrastiveDivergence):
    def __init__(self, input_size, nb_particles=128):
        ContrastiveDivergence.__init__(self)
        self.is_PCD = True
        self.particles = theano.shared(np.zeros((nb_particles, input_size), dtype=theano.config.floatX))

    def __call__(self, model, chain_start, cdk=1):
        chain_start = self.particles[:chain_start.shape[0]]
        chain_end, updates = ContrastiveDivergence.__call__(self, model, chain_start, cdk)

        # Update particles
        updates[self.particles] = T.set_subtensor(chain_start, chain_end)

        return chain_end, updates
