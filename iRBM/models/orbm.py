import numpy as np

import theano
import theano.tensor as T

from iRBM.misc.utils import logsumexp
from iRBM.models.rbm import RBM


class oRBM(RBM):
    """Ordered Restricted Boltzmann Machine (oRBM)  """
    def __init__(self,
                 input_size,
                 hidden_size,
                 beta=1,
                 penalty="softplus_bi",
                 *args, **kwargs):

        RBM.__init__(self, input_size, hidden_size, *args, **kwargs)

        self.penalty = penalty
        self.beta = theano.shared(np.array(beta, dtype=theano.config.floatX), name="beta")

    def E(self, v, h, z):
        energy = -T.dot(v, self.c)
        energy += (-T.dot(T.dot(h[:, :z], self.W[:z, :]), v.T).T - T.dot(h[:, :z], self.b[:z])).T

        if self.penalty == "softplus_bi":
            energy += T.sum(self.beta*T.log(1+T.exp(self.b[:z])))  # Add penality term
        elif self.penalty == "softplus0":
            energy += T.sum(self.beta*T.log(1+T.exp(0)))  # Add penality term
        else:
            raise NameError("Invalid penalty term")

        return energy

    def free_energy(self, v):
        """ Marginalization over hidden units"""
        free_energy = -T.dot(v, self.c) - logsumexp(self.log_z_given_v(v), axis=1)  # Sum over z'
        return free_energy

    def log_z_given_v(self, v):
        Wx_plusb = T.dot(v, self.W.T) + self.b

        energies = T.nnet.softplus(Wx_plusb)  # Sum over h'

        if self.penalty == "softplus_bi":
            energies -= self.beta*T.nnet.softplus(self.b)  # Add penality term
        elif self.penalty == "softplus0":
            energies -= self.beta*T.nnet.softplus(0)  # Add penality term
        else:
            raise NameError("Invalid penalty term")

        energies = T.cumsum(energies, axis=1)   # Cumsum over z
        return energies

    def pdf_z_given_v(self, v):
        log_z_given_v = self.log_z_given_v(v)
        prob_z_given_v = T.nnet.softmax(log_z_given_v)  # If 2D, softmax is perform along axis=1.
        return prob_z_given_v

    def icdf_z_given_v(self, v):
        return T.cumsum(self.pdf_z_given_v(v)[:, ::-1], axis=1)[:, ::-1]

    def sample_zmask_given_v(self, v):
        p = self.theano_rng.multinomial(pvals=self.pdf_z_given_v(v), dtype=theano.config.floatX)
        return T.cumsum(p[:, ::-1], axis=1)[:, ::-1]

    def sample_h_given_v(self, v):
        hidden_size = self.b.shape[0]

        z_mask = self.sample_zmask_given_v(v)

        Wx_plusb = T.dot(v, self.W.T) + self.b
        #Wx_plusb = z_mask * (T.dot(v, self.W.T) + self.b)
        prob_h = 1 / (1 + T.exp(-Wx_plusb))
        prob_h_nil = 1 / (1 + T.exp(Wx_plusb))

        prob = T.stack(prob_h_nil, prob_h)
        prob = T.reshape(prob, (2, v.shape[0]*hidden_size)).T  # Needs to reshape because right now Theano GPU's multinomial supports only pvals.ndim==2 and n==1.

        h_sample = self.theano_rng.multinomial(n=1, pvals=prob, dtype=theano.config.floatX)
        h_sample = T.dot(h_sample, np.array([0, 1], dtype=theano.config.floatX))

        h_sample = T.reshape(h_sample, (v.shape[0], hidden_size))  # Needs to reshape because right now Theano GPU's multinomial supports only pvals.ndim==2 and n==1.
        return z_mask * h_sample

    def sample_v_given_h(self, h):
        Wh_plusc = T.dot(h, self.W) + self.c  # Since h \in {0,1}
        v_mean = T.nnet.sigmoid(Wh_plusc)
        v_sample = self.theano_rng.binomial(size=v_mean.shape, n=1, p=v_mean, dtype=theano.config.floatX)
        return v_sample

    def __getstate__(self):
        state = {}
        state.update(RBM.__getstate__(self))
        state['oRBM_version'] = 1

        # Hyper parameters
        state['beta'] = self.beta.get_value()
        state['penalty'] = self.penalty

        return state

    def __setstate__(self, state):
        RBM.__setstate__(self, state)

        # Hyper parameters
        self.beta = theano.shared(state['beta'], name="beta")
        self.penalty = state['penalty']

    def marginalize_over_v(self, h, z):
        energy = 0
        energy += T.dot(h[:, :z], self.b[:z])

        if self.penalty == "softplus_bi":
            energy -= T.sum(self.beta*T.log(1 + T.exp(self.b[:z])), keepdims=True)
        elif self.penalty == "softplus0":
            energy -= T.sum(self.beta*T.log(1 + T.exp(0)), keepdims=True)
        else:
            raise NameError("Invalid penalty term")

        energy += T.sum(T.log(1 + T.exp(T.dot(h[:, :z], self.W[:z])+self.c)), axis=1)
        return energy

    def marginalize_over_v_z(self, h):
        # energy = \sum_{i=1}^{|h|} h_i*b_i - \beta * ln(1 + e^{b_i})

        # In theory should use the following line
        # energy = (h * self.b).T
        # However, when there is broadcasting, the Theano element-wise multiplication between np.NaN and 0 is 0 instead of np.NaN!
        # so we use T.tensordot and T.diagonal instead as a workaround!
        # See Theano issue #3848 (https://github.com/Theano/Theano/issues/3848)
        energy = T.tensordot(h, self.b, axes=0)
        energy = T.diagonal(energy, axis1=1, axis2=2).T

        if self.penalty == "softplus_bi":
            energy = energy - self.beta * T.log(1 + T.exp(self.b))[:, None]

        elif self.penalty == "softplus0":
            energy = energy - self.beta * T.log(1 + T.exp(0))[:, None]

        else:
            raise NameError("Invalid penalty term")

        energy = T.set_subtensor(energy[(T.isnan(energy)).nonzero()], 0)  # Remove NaN
        energy = T.sum(energy, axis=0, keepdims=True).T

        ener = T.tensordot(h, self.W, axes=0)
        ener = T.diagonal(ener, axis1=1, axis2=2)
        ener = T.set_subtensor(ener[(T.isnan(ener)).nonzero()], 0)
        ener = T.sum(ener, axis=2) + self.c[None, :]
        ener = T.sum(T.log(1 + T.exp(ener)), axis=1, keepdims=True)

        return -(energy + ener)

    def get_base_rate(self, base_rate_type="uniform"):
        base_rate, annealable_params = RBM.get_base_rate(self, base_rate_type)
        #annealable_params.append(self.beta)  # Seems to work better without annealing self.beta (see unit tests)

        if base_rate_type == "uniform":
            def compute_lnZ(self):
                # Since biases and weights are all 0, there are $2^input_size$ different
                #  visible neuron's states having the following energy
                #  $\sum_{z=1}^H \sum_{h \in \{0,1\}^z} \exp(-\beta z \ln(2))$
                r = T.exp((1-self.beta) * T.log(2))  # Ratio of a geometric serie
                lnZ = T.log((r - r**(self.hidden_size+1)) / (1-r))
                return (self.input_size * T.log(2) +  # ln(2^input_size)
                        lnZ)  # $ln( \sum_{z=1}^H \sum_{h \in \{0,1\}^z} \exp(-\beta z \ln(2)) )$

        elif base_rate_type == "c":
            def compute_lnZ(self):
                # Since the hidden biases (but not the visible ones) and the weights are all 0
                r = T.exp((1-self.beta) * T.log(2))  # Ratio of a geometric serie
                lnZ = T.log((r - r**(self.hidden_size+1)) / (1-r))
                return (lnZ +  # $ln( \sum_{z=1}^H \sum_{h \in \{0,1\}^z} \exp(-\beta z \ln(2)) )$
                        T.sum(T.nnet.softplus(self.c)))

        elif base_rate_type == "b":
            raise NotImplementedError()

        import types
        base_rate.compute_lnZ = types.MethodType(compute_lnZ, base_rate)

        return base_rate, annealable_params
