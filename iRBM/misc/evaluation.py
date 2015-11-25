from __future__ import division

import numpy as np

import theano
import theano.tensor as T


def build_average_nll(model):
    X = T.matrix('input')
    ln_Z = T.scalar('ln_Z')
    log_p_xs = -model.free_energy(X) - ln_Z
    return theano.function([X, ln_Z], -log_p_xs.mean())


def build_avg_stderr_nll2(model, factor=1.96):
    X = T.matrix('input')
    ln_Z = T.scalar('ln_Z')
    log_p_xs = -model.free_energy(X) - ln_Z
    avg_nll = -log_p_xs.mean()
    stderr_nll = factor*(log_p_xs.std()/T.sqrt(X.shape[0]))
    return theano.function([X, ln_Z], [avg_nll, stderr_nll])


def build_avg_stderr_nll(model, factor=1.96):
    X = T.matrix('input')
    ln_Z = T.scalar('ln_Z')
    log_p_xs = -model.free_energy(X) - ln_Z
    f = theano.function([X, ln_Z], log_p_xs)

    def _process_in_batch(X, ln_Z):
        # Try different size of batch size.
        for batch_size in np.linspace(len(X), 1, 11):
            batch_size = int(np.ceil(batch_size))
            log_p_xs = np.nan * np.ones(len(X), dtype=theano.config.floatX)
            try:
                for i in range(0, len(X), batch_size):
                    log_p_xs[i:i+batch_size] = f(X[i:i+batch_size], ln_Z)

                avg_nll = -log_p_xs.mean()
                stderr_nll = factor*(log_p_xs.std()/np.sqrt(len(X)))
                return avg_nll, stderr_nll

            except MemoryError:
                # Probably not enough memory on GPU
                pass

            print "*An error occured while computing NLL. Will try a smaller batch size to compute the NLL."

        raise RuntimeError("Cannot find a suitable batch size to compute the NLL. Try using CPU instead.")

    return _process_in_batch


def build_average_free_energy(model):
    V = T.matrix('input')
    Fv = model.free_energy(V)
    return theano.function([V], Fv.mean())
