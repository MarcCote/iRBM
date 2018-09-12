# -*- coding: utf-8 -*-

import numpy as np

import os
import shutil
from os.path import join as pjoin

import theano
import theano.tensor as T
from theano import config

from iRBM.misc import utils


BETAS = np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]


def _compute_AIS_samples(model, M=100, betas=BETAS):
    """
    ref: Salakhutdinov & Murray (2008), On the quantitative analysis of deep belief networks
    """
    model.batch_size = M  # Will be executing `M` AIS's runs in parallel.

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

        del params

        return log_pk-log_pk_minus_1, updates

    # Base rate with same visible biases as the model
    base_rate, annealable_params = model.get_base_rate("c")

    betas = theano.shared(value=betas.astype(config.floatX), name="Betas", borrow=True)

    k = T.iscalar('k')
    v = theano.shared(np.zeros((M, model.input_size), dtype=config.floatX))

    sym_log_w_ais_k, updates = _log_annealed_importance_sample(v, k, betas, annealable_params)
    log_annealed_importance_sample = theano.function([k],
                                                     sym_log_w_ais_k,
                                                     updates=updates)

    lnZ_trivial = base_rate.compute_lnZ().eval()

    # Will be executing M AIS's runs.
    last_sample_chain = np.zeros((M, model.input_size), dtype=config.floatX)
    M_log_w_ais = np.zeros(M, dtype=np.float64)

    # First sample V0
    h0 = base_rate.sample_h_given_v(T.zeros((M, model.input_size), dtype=config.floatX))
    v0 = base_rate.sample_v_given_h(h0).eval()
    v.set_value(v0)  # Set initial v for AIS

    # Iterate through all betas (temperature parameter)
    for k in xrange(1, len(betas.get_value())):
        M_log_w_ais += log_annealed_importance_sample(k)

        # Keep samples generated using AIS.
        last_sample_chain = v.get_value()

    M_log_w_ais += lnZ_trivial
    return {"M_log_w_ais": M_log_w_ais,
            "last_sample_chain": last_sample_chain,
            "lnZ_trivial": lnZ_trivial}


def _compute_AIS(model, M=100, betas=BETAS, batch_size=None, seed=1234, ais_working_dir=".", force=False):
    ais_results_json = pjoin(ais_working_dir, "ais_results.part.json")

    if batch_size is None:
        batch_size = M

    # Will be executing M AIS's runs.
    last_sample_chain = np.zeros((M, model.input_size), dtype=config.floatX)
    M_log_w_ais = np.zeros(M, dtype=np.float64)

    model.set_rng_seed(seed)

    ais_results = {}
    if os.path.isfile(ais_results_json) and not force:
        print "Resuming AIS using info from {}".format(ais_results_json)
        ais_results = utils.load_dict_from_json_file(ais_results_json)
        M_log_w_ais = ais_results['M_log_w_ais']
        last_sample_chain = ais_results['last_sample_chain']
        lnZ_trivial = ais_results['lnZ_trivial']

    # Iterate through all AIS runs.
    for i in range(0, M, batch_size):
        if i <= ais_results.get('batch_id', -1):
            continue

        model.set_rng_seed(seed+i)
        actual_size = min(M - i, batch_size)
        print "AIS run: {}/{} (using batch size of {})".format(i, M, batch_size)
        ais_partial_results = _compute_AIS_samples(model, M=actual_size, betas=betas)

        M_log_w_ais[i:i+batch_size] = ais_partial_results['M_log_w_ais']
        last_sample_chain[i:i+batch_size] = ais_partial_results['last_sample_chain']
        lnZ_trivial = ais_partial_results['lnZ_trivial']

        # Save partial results
        if os.path.isfile(ais_results_json):
            shutil.copy(ais_results_json, ais_results_json[:-4] + "old.json")

        ais_results = {'batch_id': i,
                       'M': M,
                       'batch_size': batch_size,
                       'last_sample_chain': last_sample_chain,
                       'M_log_w_ais': M_log_w_ais,
                       'lnZ_trivial': lnZ_trivial}
        utils.save_dict_to_json_file(ais_results_json, ais_results)

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
    std_lnZ = np.r_[np.nan, std_lnZ]  # Standard deviation of only one sample does not exist.

    return {"logcummean_Z": logcummean_Z.astype(config.floatX),
            "logcumstd_Z_down": logcumstd_Z_down.astype(config.floatX),
            "logcumstd_Z_up": logcumstd_Z_up.astype(config.floatX),
            "logcumstd_Z": logstd_AIS.astype(config.floatX),
            "M_log_w_ais": M_log_w_ais,
            "lnZ_trivial": lnZ_trivial,
            "std_lnZ": std_lnZ,
            "last_sample_chain": last_sample_chain,
            "batch_size": batch_size,
            "seed": seed,
            "nb_temperatures": len(betas),
            "nb_samples": M
            }


def compute_AIS(model, M=100, betas=BETAS, seed=1234, ais_working_dir=".", force=False):
    # Try different size of batch size.
    batch_size = M
    while batch_size >= 1:
        "Computing AIS using batch size of {}".format(batch_size)
        try:
            return _compute_AIS(model, M=M, betas=betas, batch_size=int(batch_size), seed=seed, ais_working_dir=ais_working_dir, force=force)
        except MemoryError as e:
            print e
            # Probably not enough memory on GPU
            pass
        except ValueError as e:
            print e
            # Probably because of the limited Multinomial op
            pass

        print "*An error occured while computing AIS. Will try a smaller batch size to compute AIS."
        batch_size = batch_size // 2

    raise RuntimeError("Cannot find a suitable batch size to compute AIS. Try using CPU instead.")


def estimate_lnZ(rbm, M=100, betas=np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]):
    return compute_AIS(rbm, M, betas)['logcummean_Z'][-1]


def estimate_lnZ_with_std(rbm, M=100, betas=np.r_[np.linspace(0, 0.5, num=500), np.linspace(0.5, 0.9, num=4000), np.linspace(0.9, 1, num=10000)]):
    info = compute_AIS(rbm, M, betas)
    return info['logcummean_Z'][-1], (info['logcumstd_Z_down'][-1], info['logcumstd_Z_up'][-1]), (info['std_lnZ'][-1], info['std_lnZ'][-1])
