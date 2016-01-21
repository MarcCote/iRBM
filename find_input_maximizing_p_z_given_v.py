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

import theano

from iRBM.misc import utils
from iRBM.misc import dataset

from iRBM.misc.utils import Timer

import matplotlib.pyplot as plt
from iRBM.misc import vizu


def buildArgsParser():
    DESCRIPTION = ("Script to sample from an RBM-like model.")
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('--start', type=int, help='$a$ of the first bucket. Default: 0', default=0)
    p.add_argument('--end', type=int, help='$b$ of the last bucket. Default: max', default=None)
    p.add_argument('--bucket-size', type=int, help='size of the bucket i.e. $b-a$. Default: 12', default=12)

    # # General options (optional)
    # general = p.add_argument_group("General arguments")
    # general.add_argument('--view', action='store_true',
    #                      help='display the samples.')
    # general.add_argument('--save', action='store_true',
    #                      help='save the samples.')
    # general.add_argument('--out', metavar='FILE', type=str,
    #                      help='file where samples will be saved. Default=samples.npz', default="samples.npy")

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Check that a least one of --view or --save has been given.
    #if not args.view and not args.save:
    #    parser.error("At least one the following options must be chosen: --view or --save")

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

        if hyperparams["dataset"] == "binarized_mnist":
            image_shape = (28, 28)
        elif hyperparams["dataset"] == "caltech101_silhouettes28":
            image_shape = (28, 28)
        else:
            raise ValueError("Unknown dataset: {0}".format(hyperparams["dataset"]))

    with Timer("Loading model"):
        if hyperparams["model"] == "rbm":
            raise ValueError("RBM doesn't have a p(z|v) distribution.")
        elif hyperparams["model"] == "orbm":
            from iRBM.models.orbm import oRBM
            model_class = oRBM
        elif hyperparams["model"] == "irbm":
            from iRBM.models.irbm import iRBM
            model_class = iRBM

        # Load the actual model.
        model = model_class.load(pjoin(experiment_path, "model.pkl"))

    with Timer("Building function p(z|v)"):
        v = testset.symb_inputs
        pdf_z_given_v = theano.function([v], model.pdf_z_given_v(v))

    min_z = args.start
    max_z = model.hidden_size if args.end is None else args.end
    size = args.bucket_size
    buckets = np.arange(min_z, max_z, size)
    nb_buckets = len(buckets)

    inputs = testset.inputs.get_value()
    probs = pdf_z_given_v(inputs)

    # plt.figure()
    # plt.plot(probs.T)
    # plt.title("p(z|v) for all inputs in the testset")
    # plt.xlabel("z")
    # plt.ylabel("p(z|v)")

    topk = 10
    images = np.zeros((nb_buckets*topk, int(np.prod(image_shape))))
    images_dummy = np.zeros((nb_buckets, int(np.prod(image_shape))))
    for i, start in enumerate(buckets):
        bucket_probs = np.sum(probs[:, start:start+size], axis=1)
        indices = np.argsort(bucket_probs)[::-1]

        for j in range(topk)[::-1]:
            images[j*nb_buckets + i] = inputs[indices[j]]

        # Dummy images are used to proportionally represent, via their intensity, the mean p(a <= z < b|v) of the top-k inputs.
        images_dummy[i] = np.mean(bucket_probs[indices][:topk])

    # Prepend the dummy images so they are displayed on the first row.
    #images = np.r_[images_dummy, images]
    #data = vizu.concatenate_images(images, shape=image_shape, dim=(topk+1, nb_buckets), border_size=0, clim=(0, 1))
    data = vizu.concatenate_images(images, shape=image_shape, dim=(topk, nb_buckets), border_size=0, clim=(0, 1))

    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(24, 9))
    #ax1.set_title("Top-{} inputs maximizing $p(z|\\mathbf{{v}})$ for different values of $z$".format(topk), fontsize=20)
    xticks = image_shape[1]*np.arange(nb_buckets)+image_shape[1]//2
    ax1.plot(xticks, images_dummy[:, 0], linewidth=2)
    ax1.set_ylabel("Avg. $p(a\\leq z < b|\\mathbf{v})$ of " + "top-{}".format(topk), fontsize=20)
    ax1.set_ylim(0, 1)
    ax1.set_adjustable('box-forced')

    ax2.imshow(data, cmap=plt.cm.gray, interpolation='nearest', origin="upper")
    ax2.set_ylabel("Top-{} inputs".format(topk), fontsize=20)
    ax2.set_yticks(image_shape[1]*np.arange(topk)+image_shape[1]/2.)
    ax2.set_yticklabels(map(str, range(1, topk+1)[::-1]))
    ax2.set_anchor('N')
    #ax2.set_ylim(0, image_shape[1]*topk)

    xticks = image_shape[1]*np.arange(nb_buckets+1)
    xticks_labels = map(str, buckets) + [str(buckets[-1]+size)]
    ax2.set_xlabel("$a\\leq z < b$", fontsize=20)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks_labels, rotation=45)
    ax2.set_xlim(min(xticks), max(xticks))
    ax2.set_adjustable('box-forced')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0.)
    #plt.setp(ax1.get_xticklabels(), visible=False)

    f.tight_layout()
    plt.savefig("topk_prob_z_given_x.png", dpi=300, bbox_inches='tight')
    print "Saving to ./topk_prob_z_given_x.png"
    #plt.show()
    return

    plt.figure()

    plt1 = plt.subplot(2, 1, 2)
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.ylabel("Top-{}".format(topk), fontsize=20)
    #plt.yticks(image_shape[1]*np.arange(topk+1)+image_shape[1]/2., ["Intensity"] + map(str, range(1, topk+1)[::-1]))
    plt.yticks(image_shape[1]*np.arange(topk)+image_shape[1]/2., map(str, range(1, topk+1)[::-1]))
    plt.xlabel("$a\\leq z < b$", fontsize=20)

    xticks = image_shape[1]*np.arange(nb_buckets+1)
    xticks_labels = map(str, buckets) + [str(buckets[-1]+size)]
    plt.xticks(xticks, xticks_labels, rotation=45)
    plt.xlim(min(xticks), max(xticks))
    #plt.xlim(min(xticks), max(xticks))

    plt.subplot(2, 1, 1, sharex=plt1)
    plt.title("Top-{} inputs maximizing $p(z|\\mathbf{{v}})$ for different values of $z$".format(topk), fontsize=20)
    #x = buckets+size//2
    xticks = image_shape[1]*np.arange(nb_buckets)+image_shape[1]//2
    plt.plot(xticks, images_dummy[:, 0])
    plt.ylabel("Mean $p(a\\leq z < b|\\mathbf{v})$", fontsize=20)
    #plt.xticks(xticks, map(str, buckets) + [str(buckets[-1]+size)], rotation=45)
    #plt.xticks(xticks, [])
    #plt.xlim(min(xticks)-image_shape[1]//2, max(xticks)+image_shape[1]//2)


    plt.subplots_adjust(hspace=0.001, left=0., right=1., top=1., bottom=0.)
    plt.tight_layout()
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
