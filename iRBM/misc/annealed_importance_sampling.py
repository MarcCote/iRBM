# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
from theano import config


def _compute_AIS(model, M=100, betas=np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)], batch_size=None):
    """
    ref: Salakhutdinov & Murray (2008), On the quantitative analysis of deep belief networks
    """
    if batch_size is None:
        batch_size = M

    model.batch_size = batch_size  # Will be executing `batch_size` AIS's runs in parallel.

    def _log_annealed_importance_sample(v, k, betas, annealable_params):
        beta_k = betas[k]
        beta_k_minus_1 = betas[k-1]

        # Important to backup model's parameters as we modify them
        params = [(param.name, param.clone()) for param in annealable_params]

        # Set `param * beta_k_minus_1`
        for name, param in params:
            setattr(model, name, param * beta_k_minus_1)

        log_pk_minus_1 = -model.free_energy(v)

        # Set `param * beta_k`
        for name, param in params:
            setattr(model, name, param * beta_k)

        log_pk = -model.free_energy(v)

        updates = {}
        h = model.sample_h_given_v(v)
        updates[v] = model.sample_v_given_h(h)

        # Restore original parameters of the model.
        for name, param in params:
            setattr(model, name, param)

        return log_pk-log_pk_minus_1, updates

    # Base rate with same visible biases as the model
    base_rate, annealable_params = model.get_base_rate("c")

    betas = theano.shared(value=betas.astype(config.floatX), name="Betas", borrow=True)

    k = T.iscalar('k')
    v = theano.shared(np.zeros((batch_size, model.input_size), dtype=config.floatX))

    sym_log_w_ais_k, updates = _log_annealed_importance_sample(v, k, betas, annealable_params)
    log_annealed_importance_sample = theano.function([k],
                                                     sym_log_w_ais_k,
                                                     updates=updates)

    lnZ_trivial = base_rate.compute_lnZ().eval()

    # Will be executing M AIS's runs.
    M_log_w_ais = np.zeros(M, dtype=np.float64)

    # Iterate through all betas (temperature parameter)
    for i in range(0, M, batch_size):
        print "AIS run: {}/{} (using batch size of {})".format(i, M, batch_size)

        h0 = base_rate.sample_h_given_v(T.zeros((batch_size, model.input_size), dtype=config.floatX))
        v0 = base_rate.sample_v_given_h(h0).eval()
        v.set_value(v0)  # Set initial v for AIS

        for k in xrange(1, len(betas.get_value())):
            M_log_w_ais[i:i+batch_size] += log_annealed_importance_sample(k)

    M_log_w_ais += lnZ_trivial

    # We compute the mean of the estimated `r_AIS`
    Ms = np.arange(1, M+1)
    log_sum_w_ais = np.logaddexp.accumulate(M_log_w_ais)
    logcummean_Z = log_sum_w_ais - np.log(Ms)

    # We compute the standard deviation of the estimated `r_AIS`
    logstd_AIS = np.zeros_like(M_log_w_ais)
    for k in Ms[1:]:
        m = np.max(M_log_w_ais[:k])
        logstd_AIS[k-1] = np.log(np.std(np.exp(M_log_w_ais[:k]-m), ddof=1)) - np.log(np.sqrt(k))
        logstd_AIS[k-1] += m

    logstd_AIS[0] = np.nan  # Standard deviation of only one sample does not exist.

    # The authors report AIS error using ln(Ẑ ± 3\sigma)
    m = max(np.nanmax(logstd_AIS), np.nanmax(logcummean_Z))
    logcumstd_Z_up = np.log(np.exp(logcummean_Z-m) + 3*np.exp(logstd_AIS-m)) + m - logcummean_Z
    logcumstd_Z_down = -(np.log(np.exp(logcummean_Z-m) - 3*np.exp(logstd_AIS-m)) + m) + logcummean_Z

    # Compute the standard deviation of ln(Z)
    std_lnZ = np.array([np.std(M_log_w_ais[:k], ddof=1) for k in Ms[1:]])
    std_lnZ[0] = np.nan  # Standard deviation of only one sample does not exist.

    return {"logcummean_Z": logcummean_Z.astype(config.floatX),
            "logcumstd_Z_down": logcumstd_Z_down.astype(config.floatX),
            "logcumstd_Z_up": logcumstd_Z_up.astype(config.floatX),
            "logcumstd_Z": logstd_AIS.astype(config.floatX),
            "M_log_w_ais": M_log_w_ais,
            "lnZ_trivial": lnZ_trivial,
            "std_lnZ": std_lnZ,
            "last_sample_chain": v.get_value()
            }


def compute_AIS(model, M=100, betas=np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]):
    # Try different size of batch size.
    for batch_size in np.linspace(M, 1, 11):
        try:
            return _compute_AIS(model, M=M, betas=betas, batch_size=int(batch_size))
        except MemoryError:
            # Probably not enough memory on GPU
            pass
        except ValueError:
            # Probably because of the limited Multinomial op
            pass

        print "*An error occured while computing AIS. Will try a smaller batch size to compute AIS."

    raise RuntimeError("Cannot find a suitable batch size to compute AIS. Try using CPU instead.")


def estimate_lnZ(rbm, M=100, betas=np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]):
    return compute_AIS(rbm, M, betas)['logcummean_Z'][-1]


def estimate_lnZ_with_std(rbm, M=100, betas=np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]):
    info = compute_AIS(rbm, M, betas)
    return info['logcummean_Z'][-1], (info['logcumstd_Z_down'][-1], info['logcumstd_Z_up'][-1]), (info['std_lnZ'][-1], info['std_lnZ'][-1])
