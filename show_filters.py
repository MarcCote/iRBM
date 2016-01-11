#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import theano
theano.config.experimental.unpickle_gpu_on_cpu = True

import numpy as np
from os.path import join as pjoin
import argparse

from iRBM.misc import utils
from iRBM.misc import dataset

from iRBM.misc.utils import Timer

import pylab as plt
from iRBM.misc import vizu


def buildArgsParser():
    DESCRIPTION = ("Script to show filters learned by an RBM-like model.")
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('name', type=str, help='name/path of the experiment.')

    p.add_argument('--contrast', type=float, help='higher the value, higher the contrast. Default=1.', default=1)
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

    if hyperparams["dataset"] == "binarized_mnist":
        image_shape = (28, 28)
    elif hyperparams["dataset"] == "caltech101_silhouettes28":
        image_shape = (28, 28)
    else:
        raise ValueError("Unknown dataset: {0}".format(hyperparams["dataset"]))

    weights = model.W.get_value()
    clim = (weights.min(), weights.max())
    data = vizu.concatenate_images(args.contrast*weights, shape=image_shape, border_size=1, clim=clim)
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
