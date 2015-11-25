#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from os.path import join as pjoin
import argparse

from iRBM.misc import utils
from iRBM.misc import dataset

from iRBM.misc.utils import Timer

from iRBM.misc.annealed_importance_sampling import compute_AIS
from iRBM.misc.evaluation import build_average_free_energy, build_avg_stderr_nll
from collections import namedtuple

NLL = namedtuple('NLL', ['avg', 'stderr'])


def compute_AvgFv(model, *datasets):
    avg_fv = build_average_free_energy(model)
    return map(avg_fv, datasets)


def compute_lnZ(model, nb_chains, temperatures):
    ais_results = compute_AIS(model, M=nb_chains, betas=temperatures)
    lnZ_est = ais_results['logcummean_Z'][-1]
    lnZ_down = ais_results['logcumstd_Z_down'][-1]
    lnZ_up = ais_results['logcumstd_Z_up'][-1]
    return lnZ_est, lnZ_down, lnZ_up


def compute_AvgStderrNLL(model, lnZ, *datasets):
    avg_stderr_nll = build_avg_stderr_nll(model)
    datasets = map(lambda d: d.inputs.get_value(), datasets)
    nlls = map(avg_stderr_nll, datasets, [lnZ]*len(datasets))
    # Convert list of ndarrays to the NLL namedtuple.
    return [NLL(float(nll[0]), float(nll[1])) for nll in nlls]


def buildArgsParser():
    DESCRIPTION = ("Script to evaluate an RBM-like model using "
                   "annealed importance sampling (AIS) method to approximate the partition function.")
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('name', type=str, help='name/path of the experiment.')

    ais = p.add_argument_group("AIS arguments")
    ais.add_argument('--nb-samples', metavar='M', type=int,
                     help='use M samples in AIS. Default=5000', default=5000)
    ais.add_argument('--nb-temperatures', metavar='N', type=int,
                     help='AIS will be performed using N temperatures between [0,1]. Default 100000.', default=100000)

    lnZ = p.add_argument_group("Partition function (lnZ) informations")
    lnZ.add_argument('--lnZ', metavar=("lnZ", "lnZ_down", "lnZ_up"), type=float, nargs=3,
                     help='use this information (i.e. lnZ lnZ_down lnZ_up) about the partition function instead of approximating it with AIS.')

    p.add_argument('-f', '--force', action='store_true', help='Overwrite existing `result.json`')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Get experiment folder
    experiment_path = args.name
    if not os.path.isdir(experiment_path):
        # If not a directory, it must be the name of the experiment.
        experiment_path = pjoin(".", "experiments", args.name)

    if not os.path.isdir(experiment_path):
        parser.error('Cannot find experiment: {0}!'.format(args.name))

    if not os.path.isfile(pjoin(experiment_path, "model.pkl")):
        parser.error('Cannot find model for experiment: {0}!'.format(experiment_path))

    if not os.path.isfile(pjoin(experiment_path, "hyperparams.json")):
        parser.error('Cannot find hyperparams for experiment: {0}!'.format(experiment_path))

    result_file = pjoin(experiment_path, "result.json")
    if os.path.isfile(result_file) and not args.force:
        parser.error('{0} already exists. Use --force to overwrite it.'.format(result_file))

    # Load experiments hyperparameters
    hyperparams = utils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))

    with Timer("Loading dataset"):
        trainset, validset, testset = dataset.load(hyperparams['dataset'], hyperparams.get('dataset_percent', 1.))
        print " (data: {:,}; {:,}; {:,}) ".format(len(trainset), len(validset), len(testset)),

    with Timer("Loading model"):
        if hyperparams["model"] == "rbm":
            from iRBM.models.rbm import RBM
            model_class = RBM
        elif hyperparams["model"] == "orbm":
            from iRBM.models.orbm import oRBM
            model_class = oRBM
        elif hyperparams["model"] == "irbm":
            from iRBM.models.irbm import iRBM
            model_class = iRBM

        # Load the actual model.
        model = model_class.load(pjoin(experiment_path, "model.pkl"))

    if args.lnZ is None:
        with Timer("Estimating model's partition function with AIS({0}) and {1} temperatures.".format(args.nb_samples, args.nb_temperatures)):
            lnZ, lnZ_down, lnZ_up = compute_lnZ(model, nb_chains=args.nb_samples, temperatures=np.linspace(0, 1, args.nb_temperatures))
            lnZ_down = lnZ - lnZ_down
            lnZ_up = lnZ + lnZ_up
    else:
        lnZ, lnZ_down, lnZ_up = args.lnZ

    print "-> lnZ: {lnZ_down} <= {lnZ} <= {lnZ_up}".format(lnZ_down=lnZ_down, lnZ=lnZ, lnZ_up=lnZ_up)

    with Timer("\nComputing average NLL on {0} using lnZ={1}.".format(hyperparams['dataset'], lnZ)):
        NLL_train, NLL_valid, NLL_test = compute_AvgStderrNLL(model, lnZ, trainset, validset, testset)

    print "Avg. NLL on trainset: {:.2f} ± {:.2f}".format(NLL_train.avg, NLL_train.stderr)
    print "Avg. NLL on validset: {:.2f} ± {:.2f}".format(NLL_valid.avg, NLL_valid.stderr)
    print "Avg. NLL on testset:  {:.2f} ± {:.2f}".format(NLL_test.avg, NLL_test.stderr)

    # Save results JSON file.
    result = {'lnZ': float(lnZ),
              'lnZ_down': float(lnZ_down),
              'lnZ_up': float(lnZ_up),
              'trainset': [float(NLL_train.avg), float(NLL_train.stderr)],
              'validset': [float(NLL_valid.avg), float(NLL_valid.stderr)],
              'testset': [float(NLL_test.avg), float(NLL_test.stderr)],
              }
    utils.save_dict_to_json_file(result_file, result)

if __name__ == "__main__":
    main()
