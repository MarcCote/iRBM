from __future__ import division

import theano
import theano.tensor as T


def build_average_nll(model):
    X = T.matrix('input')
    ln_Z = T.scalar('ln_Z')
    log_p_xs = -model.free_energy(X) - ln_Z
    return theano.function([X, ln_Z], -log_p_xs.mean())


def build_avg_stderr_nll(model, factor=1.96):
    X = T.matrix('input')
    ln_Z = T.scalar('ln_Z')
    log_p_xs = -model.free_energy(X) - ln_Z
    avg_nll = -log_p_xs.mean()
    stderr_nll = factor*(log_p_xs.std()/T.sqrt(X.shape[0]))
    return theano.function([X, ln_Z], [avg_nll, stderr_nll])


def build_average_free_energy(model):
    V = T.matrix('input')
    Fv = model.free_energy(V)
    return theano.function([V], Fv.mean())
