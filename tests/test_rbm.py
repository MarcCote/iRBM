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
from iRBM.misc.annealed_importance_sampling import compute_AIS

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nose.tools import assert_equal, assert_raises
from nose.plugins.skip import SkipTest

import numpy.testing as npt
from numpy.testing import (assert_array_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           assert_raises,
                           run_module_suite)

from iRBM.misc.utils import logsumexp
from iRBM.misc.utils import cartesian


class Test_RBM(unittest.TestCase):
    def setUp(self):
        self.input_size = 4
        self.hidden_size = 3
        self.batch_size = 100

        rng = np.random.RandomState(42)
        self.W = rng.randn(self.hidden_size, self.input_size).astype(config.floatX)
        self.b = rng.randn(self.hidden_size).astype(config.floatX)
        self.c = rng.randn(self.input_size).astype(config.floatX)

        self.model = RBM(input_size=self.input_size,
                         hidden_size=self.hidden_size)

        self.model.W.set_value(self.W)
        self.model.b.set_value(self.b)
        self.model.c.set_value(self.c)

    def test_free_energy(self):
        v = T.matrix('v')
        h = T.matrix('h')
        logsumexp_E = theano.function([v, h], -logsumexp(-self.model.E(v, h)))

        v1 = np.random.rand(1, self.input_size).astype(config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)
        Fv = logsumexp_E(v1, H)  # Marginalization over $\bh$

        v = T.matrix('v')
        free_energy = theano.function([v], self.model.free_energy(v))
        assert_array_almost_equal(free_energy(v1), [Fv])

        v2 = np.tile(v1, (self.batch_size, 1))
        assert_array_almost_equal(free_energy(v2), [Fv]*self.batch_size)

    def test_marginalize_over_v(self):
        v = T.matrix('v')
        h = T.matrix('h')
        E = theano.function([v, h], -logsumexp(-self.model.E(v, h)))

        h1 = np.random.rand(1, self.hidden_size).astype(config.floatX)
        V = cartesian([(0, 1)] * self.input_size, dtype=config.floatX)
        expected_energy = E(V, h1)

        h = T.matrix('h')
        marginalize_over_v = theano.function([h], self.model.marginalize_over_v(h))
        assert_array_almost_equal(marginalize_over_v(h1), [expected_energy])

        h2 = np.tile(h1, (self.batch_size, 1))
        assert_array_almost_equal(marginalize_over_v(h2), [expected_energy]*self.batch_size)

    def test_compute_lnZ(self):
        v = T.matrix('v')
        h = T.matrix('h')
        lnZ = theano.function([v, h], logsumexp(-self.model.E(v, h)))

        V = cartesian([(0, 1)] * self.input_size, dtype=config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)

        lnZ_using_free_energy = theano.function([v], logsumexp(-self.model.free_energy(v)))
        assert_equal(lnZ_using_free_energy(V), lnZ(V, H))

        lnZ_using_marginalize_over_v = theano.function([h], logsumexp(-self.model.marginalize_over_v(h)))
        assert_almost_equal(lnZ_using_marginalize_over_v(H), lnZ(V, H), decimal=6)

    def test_base_rate(self):
        # All binary combinaisons for V and H.
        V = cartesian([(0, 1)] * self.input_size, dtype=config.floatX)
        H = cartesian([(0, 1)] * self.hidden_size, dtype=config.floatX)

        base_rates = []
        # Add the uniform base rate, i.e. all parameters of the model are set to 0.
        base_rates.append(self.model.get_base_rate())
        # Add the base rate where visible biases are the ones from the model.
        base_rates.append(self.model.get_base_rate('c'))
        # Add the base rate where hidden biases are the ones from the model.
        base_rates.append(self.model.get_base_rate('b'))  # Not implemented

        for base_rate, anneable_params in base_rates:
            base_rate_lnZ = base_rate.compute_lnZ().eval().astype(config.floatX)

            brute_force_lnZ = logsumexp(-base_rate.E(V, H)).eval()
            assert_almost_equal(brute_force_lnZ.astype(config.floatX), base_rate_lnZ, decimal=6)

            theano_lnZ = logsumexp(-base_rate.free_energy(V), axis=0).eval()
            assert_almost_equal(theano_lnZ.astype(config.floatX), base_rate_lnZ, decimal=6)

            theano_lnZ = logsumexp(-base_rate.marginalize_over_v(H)).eval()
            assert_almost_equal(theano_lnZ.astype(config.floatX), base_rate_lnZ, decimal=6)

    @npt.dec.slow
    def test_binomial_from_uniform_cpu(self):
        #Test using numpy
        rng = np.random.RandomState(42)
        probs = rng.rand(10)

        seed = 1337
        nb_samples = 1000000
        rng = np.random.RandomState(seed)
        success1 = np.zeros(len(probs))
        for i in range(nb_samples):
            success1 += rng.binomial(n=1, p=probs)

        rng = np.random.RandomState(seed)
        success2 = np.zeros(len(probs))
        for i in range(nb_samples):
            success2 += (rng.rand(len(probs)) < probs).astype('int')

        success1 = success1 / nb_samples
        success2 = success2 / nb_samples

        assert_array_almost_equal(success1, success2)

        #Test using Theano's default RandomStreams
        theano_rng = RandomStreams(1337)
        rng_bin = theano_rng.binomial(size=probs.shape, n=1, p=probs, dtype=theano.config.floatX)
        success1 = np.zeros(len(probs))
        for i in range(nb_samples):
            success1 += rng_bin.eval()

        theano_rng = RandomStreams(1337)
        rng_bin = theano_rng.uniform(size=probs.shape, dtype=theano.config.floatX) < probs
        success2 = np.zeros(len(probs))
        for i in range(nb_samples):
            success2 += rng_bin.eval()

        assert_array_almost_equal(success1/nb_samples, success2/nb_samples)

        #Test using Theano's sandbox MRG RandomStreams
        theano_rng = MRG_RandomStreams(1337)
        success1 = theano_rng.binomial(size=probs.shape, n=1, p=probs, dtype=theano.config.floatX)

        theano_rng = MRG_RandomStreams(1337)
        success2 = theano_rng.uniform(size=probs.shape, dtype=theano.config.floatX) < probs

        assert_array_equal(success1.eval(), success2.eval())

    def test_gradients_auto_vs_manual(self):
        rng = np.random.RandomState(42)

        batch_size = 5
        input_size = 10

        rbm = RBM(input_size=input_size,
                  hidden_size=32,
                  CDk=1,
                  rng=np.random.RandomState(42))

        W = (rng.rand(rbm.hidden_size, rbm.input_size) > 0.5).astype(theano.config.floatX)
        rbm.W = theano.shared(value=W.astype(theano.config.floatX), name='b', borrow=True)

        b = (rng.rand(rbm.hidden_size) > 0.5).astype(theano.config.floatX)
        rbm.b = theano.shared(value=b.astype(theano.config.floatX), name='b', borrow=True)

        c = (rng.rand(rbm.input_size) > 0.5).astype(theano.config.floatX)
        rbm.c = theano.shared(value=c.astype(theano.config.floatX), name='c', borrow=True)

        params = [rbm.W, rbm.b, rbm.c]
        chain_start = T.matrix('start')
        chain_end = T.matrix('end')

        chain_start_value = (rng.rand(batch_size, input_size) > 0.5).astype(theano.config.floatX)
        chain_end_value = (rng.rand(batch_size, input_size) > 0.5).astype(theano.config.floatX)
        chain_start.tag.test_value = chain_start_value
        chain_end.tag.test_value = chain_end_value

        ### Computing gradients using automatic differentation ###
        cost = T.mean(rbm.free_energy(chain_start)) - T.mean(rbm.free_energy(chain_end))
        gparams_auto = T.grad(cost, params, consider_constant=[chain_end])

        ### Computing gradients manually ###
        h = rbm.sample_h_given_v(chain_start, return_probs=True)
        _h = rbm.sample_h_given_v(chain_end, return_probs=True)

        grad_W = (T.dot(chain_end.T, _h) - T.dot(chain_start.T, h)).T / batch_size
        grad_b = T.mean(_h - h, 0)
        grad_c = T.mean(chain_end - chain_start, 0)

        gparams_manual = [grad_W, grad_b, grad_c]
        grad_W.name, grad_b.name, grad_c.name = "grad_W", "grad_b", "grad_c"

        for gparam_auto, gparam_manual in zip(gparams_auto, gparams_manual):
            param1 = gparam_auto.eval({chain_start: chain_start_value, chain_end: chain_end_value})
            param2 = gparam_manual.eval({chain_start: chain_start_value, chain_end: chain_end_value})
            assert_array_almost_equal(param1, param2, err_msg=gparam_manual.name)


class TestAIS_RBM(unittest.TestCase):
    def setUp(self):
        self.nb_samples = 1000
        self.input_size = 10
        self.hidden_size = 14

        rng = np.random.RandomState(42)
        self.W = rng.rand(self.hidden_size, self.input_size).astype(config.floatX)
        self.b = rng.rand(self.hidden_size).astype(config.floatX)
        self.c = rng.rand(self.input_size).astype(config.floatX)

        self.betas = np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]

    def test_verify_AIS(self):
        model = RBM(input_size=self.input_size,
                    hidden_size=self.hidden_size)

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
            ais_working_dir = tempfile.mkdtemp()
            result = compute_AIS(model, M=self.nb_samples, betas=self.betas, seed=1234, ais_working_dir=ais_working_dir, force=True)
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

            print "Absolute diff:", np.abs(AIS_logZ - f_brute_force_lnZ())
            assert_almost_equal(AIS_logZ, f_brute_force_lnZ(), decimal=2)
        finally:
            shutil.rmtree(ais_working_dir)


if __name__ == '__main__':
    run_module_suite()
