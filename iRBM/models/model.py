import pickle
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from iRBM.misc.learning_rate import ConstantLearningRate
from iRBM.misc.regularization import NoRegularization


class Model():
    def __init__(self,
                 learning_rate=ConstantLearningRate(lr=1),
                 regularization=NoRegularization(),
                 rng=np.random.RandomState()):

        self.learning_rate = learning_rate
        self.regularize = regularization
        self.np_rng = rng
        self.theano_rng = RandomStreams(rng.randint(2**30))

        self.batch_size = 1

    def post_update(self, no_epoch, no_batch):
        pass

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def __getstate__(self):
        state = {}
        state['version'] = 1
        state['batch_size'] = self.batch_size

        state['learning_rate'] = self.learning_rate
        state['regularization'] = self.regularize

        # Save random generators
        state['np_rng'] = pickle.dumps(self.np_rng)
        state['theano_rng'] = pickle.dumps(self.theano_rng)

        return state

    def __setstate__(self, state):
        self.batch_size = state['batch_size']

        self.learning_rate = state['learning_rate']
        self.regularize = state['regularization']

        # Load random generators
        self.np_rng = pickle.loads(state['np_rng'])
        self.theano_rng = pickle.loads(state['theano_rng'])

    def save(self, filename):
        pickle.dump(self, open(filename, 'w'))

    @classmethod
    def load(self, filename):
        return pickle.load(open(filename, 'r'))
