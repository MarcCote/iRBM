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
    ais.add_argument('--seed', type=int,
                     help="Seed used to initialize model's random generator. Default: 1234", default=1234)

    lnZ = p.add_argument_group("Partition function (lnZ) informations")
    lnZ.add_argument('--lnZ', metavar=("lnZ", "lnZ_down", "lnZ_up"), type=float, nargs=3,
                     help='use this information (i.e. lnZ lnZ_down lnZ_up) about the partition function instead of approximating it with AIS.')

    general = p.add_argument_group("General arguments")
    general.add_argument('--view', action='store_true',
                         help='display AIS graph.')

    general.add_argument('--irbm-fixed-size', action='store_true',
                         help='when evaluating an iRBM consider it is an oRBM, i.e. the number of hidden is now fixed to $l$.')

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

        if args.irbm_fixed_size:
            # Use methods from the oRBM.
            import functools
            from iRBM.models.orbm import oRBM
            setattr(model, "get_base_rate", functools.partial(oRBM.get_base_rate, model))
            setattr(model, "pdf_z_given_v", functools.partial(oRBM.pdf_z_given_v, model))
            setattr(model, "log_z_given_v", functools.partial(oRBM.log_z_given_v, model))
            setattr(model, "free_energy", functools.partial(oRBM.free_energy, model))
            print "({} with {} fixed hidden units)".format(hyperparams["model"], model.hidden_size)

        else:
            print "({} with {} hidden units)".format(hyperparams["model"], model.hidden_size)

    # Result files.
    if args.irbm_fixed_size:
        experiment_path = pjoin(experiment_path, "irbm_fixed_size")
        try:
            os.makedirs(experiment_path)
        except:
            pass

    ais_result_file = pjoin(experiment_path, "ais_result.json")
    result_file = pjoin(experiment_path, "result.json")

    if args.lnZ is not None:
        lnZ, lnZ_down, lnZ_up = args.lnZ
    else:
        if not os.path.isfile(ais_result_file) or args.force:
            with Timer("Estimating model's partition function with AIS({0}) and {1} temperatures.".format(args.nb_samples, args.nb_temperatures)):
                ais_results = compute_AIS(model, M=args.nb_samples, betas=np.linspace(0, 1, args.nb_temperatures), seed=args.seed, ais_working_dir=experiment_path, force=args.force)
                ais_results["irbm_fixed_size"] = args.irbm_fixed_size
                utils.save_dict_to_json_file(ais_result_file, ais_results)
        else:
            print "Loading previous AIS results... (use --force to re-run AIS)"
            ais_results = utils.load_dict_from_json_file(ais_result_file)
            print "AIS({0}) with {1} temperatures".format(ais_results['nb_samples'], ais_results['nb_temperatures'])

            if ais_results['nb_samples'] != args.nb_samples:
                print "The number of samples specified ({:,}) doesn't match the one found in ais_results.json ({:,}). Aborting.".format(args.nb_samples, ais_results['nb_samples'])
                sys.exit(-1)

            if ais_results['nb_temperatures'] != args.nb_temperatures:
                print "The number of temperatures specified ({:,}) doesn't match the one found in ais_results.json ({:,}). Aborting.".format(args.nb_temperatures, ais_results['nb_temperatures'])
                sys.exit(-1)

            if ais_results['seed'] != args.seed:
                print "The seed specified ({}) doesn't match the one found in ais_results.json ({}). Aborting.".format(args.seed, ais_results['seed'])
                sys.exit(-1)

            if ais_results.get('irbm_fixed_size', False) != args.irbm_fixed_size:
                print "The option '--irbm-fixed' specified ({}) doesn't match the one found in ais_results.json ({}). Aborting.".format(args.irbm_fixed_size, ais_results['irbm_fixed_size'])
                sys.exit(-1)

        lnZ = ais_results['logcummean_Z'][-1]
        logcumstd_Z_down = ais_results['logcumstd_Z_down'][-1]
        logcumstd_Z_up = ais_results['logcumstd_Z_up'][-1]
        lnZ_down = lnZ - logcumstd_Z_down
        lnZ_up = lnZ + logcumstd_Z_up

    print "-> lnZ: {lnZ_down} <= {lnZ} <= {lnZ_up}".format(lnZ_down=lnZ_down, lnZ=lnZ, lnZ_up=lnZ_up)

    with Timer("\nComputing average NLL on {0} using lnZ={1}.".format(hyperparams['dataset'], lnZ)):
        NLL_train, NLL_valid, NLL_test = compute_AvgStderrNLL(model, lnZ, trainset, validset, testset)

    print "Avg. NLL on trainset: {:.2f} ± {:.2f}".format(NLL_train.avg, NLL_train.stderr)
    print "Avg. NLL on validset: {:.2f} ± {:.2f}".format(NLL_valid.avg, NLL_valid.stderr)
    print "Avg. NLL on testset:  {:.2f} ± {:.2f}".format(NLL_test.avg, NLL_test.stderr)

    # Save results JSON file.
    if args.lnZ is None:
        result = {'lnZ': float(lnZ),
                  'lnZ_down': float(lnZ_down),
                  'lnZ_up': float(lnZ_up),
                  'trainset': [float(NLL_train.avg), float(NLL_train.stderr)],
                  'validset': [float(NLL_valid.avg), float(NLL_valid.stderr)],
                  'testset': [float(NLL_test.avg), float(NLL_test.stderr)],
                  'irbm_fixed_size': args.irbm_fixed_size,
                  }
        utils.save_dict_to_json_file(result_file, result)

    if args.view:

        from iRBM.misc import vizu
        import matplotlib.pyplot as plt

        if hyperparams["dataset"] == "binarized_mnist":
            image_shape = (28, 28)
        elif hyperparams["dataset"] == "caltech101_silhouettes28":
            image_shape = (28, 28)
        else:
            raise ValueError("Unknown dataset: {0}".format(hyperparams["dataset"]))

        # Display AIS samples.
        data = vizu.concatenate_images(ais_results['last_sample_chain'], shape=image_shape, border_size=1, clim=(0, 1))
        plt.figure()
        plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
        plt.title("AIS samples")

        # Display AIS ~lnZ.
        plt.figure()
        plt.gca().set_xmargin(0.1)
        plt.errorbar(np.arange(ais_results['nb_samples'])+1, ais_results["logcummean_Z"],
                     yerr=[ais_results['logcumstd_Z_down'], ais_results['logcumstd_Z_up']],
                     fmt='ob', label='with std ~ln std Z')
        plt.legend()
        plt.ticklabel_format(useOffset=False, axis='y')
        plt.title("~ln mean Z for different number of AIS samples")
        plt.ylabel("~lnZ")
        plt.xlabel("# AIS samples")

        plt.show()

if __name__ == "__main__":
    main()
