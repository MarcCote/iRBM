#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from os.path import join as pjoin
import argparse

import theano

from iRBM.misc import utils

from iRBM.misc.utils import Timer

import pylab as plt
from iRBM.misc import vizu


def buildArgsParser():
    DESCRIPTION = ("Script to sample from an RBM-like model.")
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('name', type=str, help='name/path of the experiment.')

    # Sampling options
    sampling = p.add_argument_group("Sampling arguments")
    sampling.add_argument('--nb-samples', metavar='M', type=int,
                          help='number of samples. Default=16', default=16)
    sampling.add_argument('--cdk', metavar='K', type=int,
                          help='number of Gibbs steps. Default=10000', default=10000)

    sampling.add_argument('--seed', type=int,
                          help='seed used to generate random numbers. Default=1234.', default=1234)

    # General options (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('--view', action='store_true',
                         help='display the samples.')
    general.add_argument('--save', action='store_true',
                         help='save the samples.')
    general.add_argument('--out', metavar='FILE', type=str,
                         help='file where samples will be saved. Default=samples.npz', default="samples.npy")

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Check that a least one of --view or --save has been given.
    if not args.view and not args.save:
        parser.error("At least one the following options must be chosen: --view or --save")

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

    rng = np.random.RandomState(args.seed)

    # Sample from uniform
    # TODO: sample from Bernouilli distribution parametrized with visible biases
    chain_start = (rng.rand(args.nb_samples, model.input_size) > 0.5).astype(theano.config.floatX)

    with Timer("Building sampling function"):
        v0 = theano.shared(np.asarray(chain_start, dtype=theano.config.floatX))
        v1 = model.gibbs_step(v0)
        gibbs_step = theano.function([], updates={v0: v1})

    with Timer("Sampling"):
        for k in range(args.cdk):
            gibbs_step()

    samples = v0.get_value()

    if args.save:
        np.savez(args.out, samples)

    if args.view:
        if hyperparams["dataset"] == "binarized_mnist":
            image_shape = (28, 28)
        elif hyperparams["dataset"] == "caltech101_silhouettes28":
            image_shape = (28, 28)
        else:
            raise ValueError("Unknown dataset: {0}".format(hyperparams["dataset"]))

        data = vizu.concatenate_images(samples, shape=image_shape, border_size=1, clim=(0, 1))
        plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()



    # def sample(chain_start=None, nb_samples=None, cdk=1, keep=1, all_active=(0, None), model=model):
    #     import numpy as np

    #     nb_samples = len(chain_start)
    #     model.batch_size = nb_samples

    #     print("Compiling function...")
    #     v0 = theano.shared(np.asarray(chain_start, dtype=theano.config.floatX))

    #     samples = []
    #     samples.append(v0.get_value())

    #     v1 = model.gibbs_step(v0)
    #     updates = {v0: v1}
    #     gibbs_step = theano.function([], updates=updates)

    #     # Sampling with all neurons active
    #     if hasattr(model, 'beta') and all_active[0] != 0:
    #         beta = model.beta
    #         model.beta = all_active[1]
    #         updates = {v0: model.gibbs_step(v0)}
    #         gibbs_full = theano.function([], updates=updates)

    #         for i in range(all_active[0]):
    #             gibbs_full()
    #             samples.append(v0.get_value())

    #         model.beta = beta

    #     print("Sampling...")
    #     for i in range(1, cdk+1):
    #         gibbs_step()

    #         if i % keep == 0:
    #             samples.append(v0.get_value())

    #     samples = np.array(samples)
    #     return samples


if __name__ == "__main__":
    main()
