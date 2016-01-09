import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import unittest
import shutil
import tempfile

import theano
import theano.tensor as T
from theano import config

from iRBM.models.rbm import RBM
from iRBM.models.orbm import oRBM
from iRBM.misc.annealed_importance_sampling import compute_AIS

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nose.tools import assert_equal, assert_true,assert_raises
from nose.plugins.skip import SkipTest

import numpy.testing as npt
from numpy.testing import (assert_array_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           assert_raises,
                           run_module_suite)

from iRBM.misc.utils import logsumexp
from iRBM.misc.utils import cartesian


class Test_oRBM(unittest.TestCase):
    def setUp(self):
        self.input_size = 4
        self.hidden_size = 3
        self.beta = 1.01
        self.batch_size = 100

        rng = np.random.RandomState(42)
        self.W = rng.randn(self.hidden_size, self.input_size).astype(config.floatX)
        self.b = rng.randn(self.hidden_size).astype(config.floatX)
        self.c = rng.randn(self.input_size).astype(config.floatX)

        self.model = oRBM(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          beta=self.beta)

        self.model.W.set_value(self.W)
        self.model.b.set_value(self.b)
        self.model.c.set_value(self.c)

    def test_free_energy(self):
        v = T.matrix('v')
        h = T.matrix('h')
        z = T.iscalar('z')
        logsumexp_E = theano.function([v, h, z], -logsumexp(-self.model.E(v, h, z)))

        v1 = np.random.rand(1, self.input_size).astype(config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)

        energies = []
        for z in range(1, self.hidden_size+1):
            h = np.array(H[::2**(self.hidden_size-z)])
            energies.append(logsumexp_E(v1, h, z))

        Fv = -logsumexp(-np.array(energies)).eval()

        v = T.matrix('v')
        free_energy = theano.function([v], self.model.free_energy(v))
        assert_array_almost_equal(free_energy(v1), [Fv])

        v2 = np.tile(v1, (self.batch_size, 1))
        assert_array_almost_equal(free_energy(v2), [Fv]*self.batch_size)

    def test_sample_z_given_v(self):
        v = T.matrix('v')
        h = T.matrix('h')
        z = T.iscalar('z')
        E = theano.function([v, h, z], logsumexp(-self.model.E(v, h, z)))

        v1 = np.random.rand(1, self.input_size).astype(config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)

        energies = []
        for z in range(1, self.hidden_size+1):
            h = np.array(H[::2**(self.hidden_size-z)])
            energies.append(E(v1, h, z))

        probs = T.nnet.softmax(T.stack(energies))
        expected_icdf = T.cumsum(probs[:, ::-1], axis=1)[:, ::-1].eval()

        # Test inverse cdf
        v = T.matrix('v')
        icdf_z_given_v = theano.function([v], self.model.icdf_z_given_v(v))
        assert_array_almost_equal(icdf_z_given_v(v1), expected_icdf)

        batch_size = 500000
        self.model.batch_size = batch_size
        sample_zmask_given_v = theano.function([v], self.model.sample_zmask_given_v(v))
        v2 = np.tile(v1, (self.model.batch_size, 1))

        #theano.printing.pydotprint(sample_zmask_given_v)

        z_mask = sample_zmask_given_v(v2)
        # First hidden units should always be considered i.e. z_mask[:, 0] == 1
        assert_equal(np.sum(z_mask[:, 0] == 0, axis=0), 0)

        # Test that sampled masks are as expected i.e. equal expected_icdf
        freq_per_z = np.sum(z_mask, axis=0) / self.model.batch_size
        assert_array_almost_equal(freq_per_z, expected_icdf[0], decimal=3, err_msg="Tested using MC sampling, rerun it to be certain that is an error or increase 'batch_size'.")

    def test_compute_lnZ(self):
        v = T.matrix('v')
        h = T.matrix('h')
        z = T.iscalar('z')
        lnZ = theano.function([v, h, z], logsumexp(-self.model.E(v, h, z)))

        V = cartesian([(0, 1)] * self.input_size, dtype=config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)

        energies = []
        for z in range(1, self.hidden_size+1):
            hz = np.array(H[::2**(self.hidden_size-z)])
            energies.append(lnZ(V, hz, z))

        lnZ = logsumexp(np.array(energies)).eval()

        lnZ_using_free_energy = theano.function([v], logsumexp(-self.model.free_energy(v)))
        assert_almost_equal(lnZ_using_free_energy(V), lnZ, decimal=6)

        h = T.matrix('h')
        z = T.iscalar('z')
        lnZ_using_marginalize_over_v = theano.function([h, z], logsumexp(self.model.marginalize_over_v(h, z)))

        energies = []
        for z in range(1, self.hidden_size+1):
            hz = np.array(H[::2**(self.hidden_size-z)])
            energies.append(lnZ_using_marginalize_over_v(hz, z))

        assert_almost_equal(logsumexp(np.array(energies)).eval(), lnZ, decimal=6)

        # Construct Hz, a subset of H, using np.NaN as padding.
        Hz = []
        for z in range(1, self.hidden_size+1):
            hz = np.array(H[::2**(self.hidden_size-z)])
            hz[:, z:] = np.NaN
            Hz.extend(hz)

        Hz = np.array(Hz)
        assert_equal(len(Hz), np.sum(2**(np.arange(self.hidden_size)+1)))
        assert_true(len(Hz) < self.hidden_size * 2**self.hidden_size)

        lnZ_using_marginalize_over_v_z = theano.function([h], logsumexp(-self.model.marginalize_over_v_z(h)))
        assert_almost_equal(lnZ_using_marginalize_over_v_z(Hz), lnZ, decimal=6)

    def test_base_rate(self):
        # All binary combinaisons for V and H_z
        V = cartesian([(0, 1)] * self.input_size, dtype=config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)

        # Construct Hz, a subset of H, using np.NaN as padding.
        Hz = []
        for z in range(1, self.hidden_size+1):
            hz = np.array(H[::2**(self.hidden_size-z)])
            hz[:, z:] = np.NaN
            Hz.extend(hz)

        Hz = np.array(Hz)
        assert_equal(len(Hz), np.sum(2**(np.arange(self.hidden_size)+1)))
        assert_true(len(Hz) < self.hidden_size * 2**self.hidden_size)

        base_rates = []
        # Add the uniform base rate, i.e. all parameters of the model are set to 0.
        base_rates.append(self.model.get_base_rate())
        # Add the base rate where visible biases are the ones from the model.
        base_rates.append(self.model.get_base_rate('c'))
        # Add the base rate where hidden biases are the ones from the model.
        # base_rates.append(self.model.get_base_rate('b'))  # Not implemented

        for base_rate, anneable_params in base_rates:
            base_rate_lnZ = base_rate.compute_lnZ().eval().astype(config.floatX)

            v = T.matrix('v')
            h = T.matrix('h')
            z = T.iscalar('z')
            lnZ = theano.function([v, h, z], logsumexp(-base_rate.E(v, h, z)))

            energies = []
            for z in range(1, self.hidden_size+1):
                hz = np.array(H[::2**(self.hidden_size-z)])
                energies.append(lnZ(V, hz, z))

            brute_force_lnZ = logsumexp(np.array(energies)).eval()
            assert_almost_equal(brute_force_lnZ.astype(config.floatX), base_rate_lnZ, decimal=6)

            theano_lnZ = logsumexp(-base_rate.free_energy(V), axis=0).eval()
            assert_almost_equal(theano_lnZ.astype(config.floatX), base_rate_lnZ, decimal=6)

            theano_lnZ = logsumexp(-base_rate.marginalize_over_v_z(Hz)).eval()
            assert_almost_equal(theano_lnZ.astype(config.floatX), base_rate_lnZ, decimal=6)

    def test_gradients_auto_vs_manual(self):
        rng = np.random.RandomState(42)

        batch_size = 5
        input_size = 10

        model = oRBM(input_size=input_size,
                     hidden_size=32,
                     CDk=1,
                     rng=np.random.RandomState(42))

        W = rng.rand(model.hidden_size, model.input_size).astype(theano.config.floatX)
        model.W = theano.shared(value=W.astype(theano.config.floatX), name='W', borrow=True)

        b = rng.rand(model.hidden_size).astype(theano.config.floatX)
        model.b = theano.shared(value=b.astype(theano.config.floatX), name='b', borrow=True)

        c = rng.rand(model.input_size).astype(theano.config.floatX)
        model.c = theano.shared(value=c.astype(theano.config.floatX), name='c', borrow=True)

        params = [model.W, model.b, model.c]
        chain_start = T.matrix('start')
        chain_end = T.matrix('end')

        chain_start_value = (rng.rand(batch_size, input_size) > 0.5).astype(theano.config.floatX)
        chain_end_value = (rng.rand(batch_size, input_size) > 0.5).astype(theano.config.floatX)
        chain_start.tag.test_value = chain_start_value
        chain_end.tag.test_value = chain_end_value

        ### Computing gradients using automatic differentation ###
        cost = T.mean(model.free_energy(chain_start)) - T.mean(model.free_energy(chain_end))
        gparams_auto = T.grad(cost, params, consider_constant=[chain_end])

        ### Computing gradients manually ###
        h = RBM.sample_h_given_v(model, chain_start, return_probs=True)
        _h = RBM.sample_h_given_v(model, chain_end, return_probs=True)
        icdf = model.icdf_z_given_v(chain_start)
        _icdf = model.icdf_z_given_v(chain_end)

        if model.penalty == "softplus_bi":
            penalty = model.beta * T.nnet.sigmoid(model.b)
        elif self.penalty == "softplus0":
            penalty = model.beta * T.nnet.sigmoid(0)

        grad_W = (T.dot(chain_end.T, _h*_icdf) - T.dot(chain_start.T, h*icdf)).T / batch_size
        grad_b = T.mean((_h-penalty)*_icdf - (h-penalty)*icdf, axis=0)
        grad_c = T.mean(chain_end - chain_start, axis=0)

        gparams_manual = [grad_W, grad_b, grad_c]
        grad_W.name, grad_b.name, grad_c.name = "grad_W", "grad_b", "grad_c"

        for gparam_auto, gparam_manual in zip(gparams_auto, gparams_manual):
            param1 = gparam_auto.eval({chain_start: chain_start_value, chain_end: chain_end_value})
            param2 = gparam_manual.eval({chain_start: chain_start_value, chain_end: chain_end_value})
            assert_array_almost_equal(param1, param2, err_msg=gparam_manual.name)


class TestAIS_oRBM(unittest.TestCase):
    def setUp(self):
        self.nb_samples = 1000
        self.input_size = 10
        self.hidden_size = 14
        self.beta = 1.01

        rng = np.random.RandomState(42)
        self.W = rng.rand(self.hidden_size, self.input_size).astype(config.floatX)
        self.b = rng.rand(self.hidden_size).astype(config.floatX)
        self.c = rng.rand(self.input_size).astype(config.floatX)

        self.betas = np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]

    def test_verify_AIS(self):
        model = oRBM(input_size=self.input_size,
                     hidden_size=self.hidden_size,
                     beta=self.beta)

        model.W.set_value(self.W)
        model.b.set_value(self.b)
        model.c.set_value(self.c)

        # Brute force
        print "Computing lnZ using brute force (i.e. summing the free energy of all posible $v$)..."
        V = theano.shared(value=cartesian([(0, 1)] * self.input_size, dtype=config.floatX))
        brute_force_lnZ = logsumexp(-model.free_energy(V), 0)
        f_brute_force_lnZ = theano.function([], brute_force_lnZ)

        params_bak = [param.get_value() for param in model.parameters]

        print "Approximating lnZ using AIS..."
        import time
        start = time.time()

        try:
            experiment_path = tempfile.mkdtemp()
            result = compute_AIS(model, M=self.nb_samples, betas=self.betas, seed=1234, experiment_path=experiment_path, force=True)
            logcummean_Z, logcumstd_Z_down, logcumstd_Z_up = result['logcummean_Z'], result['logcumstd_Z_down'], result['logcumstd_Z_up']
            std_lnZ = result['std_lnZ']

            print "{0} sec".format(time.time() - start)

            import pylab as plt
            plt.gca().set_xmargin(0.1)
            plt.errorbar(range(1, self.nb_samples+1), logcummean_Z, yerr=[std_lnZ, std_lnZ], fmt='or')
            plt.errorbar(range(1, self.nb_samples+1), logcummean_Z, yerr=[logcumstd_Z_down, logcumstd_Z_up], fmt='ob')
            plt.plot([1, self.nb_samples], [f_brute_force_lnZ()]*2, '--g')
            plt.ticklabel_format(useOffset=False, axis='y')
            plt.show()
            AIS_logZ = logcummean_Z[-1]

            assert_array_equal(params_bak[0], model.W.get_value())
            assert_array_equal(params_bak[1], model.b.get_value())
            assert_array_equal(params_bak[2], model.c.get_value())

            print np.abs(AIS_logZ - f_brute_force_lnZ())
            assert_almost_equal(AIS_logZ, f_brute_force_lnZ(), decimal=2)
        finally:
            shutil.rmtree(experiment_path)


if __name__ == '__main__':
    run_module_suite()
