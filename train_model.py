#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from os.path import join as pjoin
import argparse
import shutil

from iRBM.training.trainer import Trainer

import iRBM.training.tasks as tasks

from iRBM.misc import utils
from iRBM.misc import dataset
from iRBM.models import model_factory, irbm

from iRBM.misc.utils import Timer


DATASETS = ["binarized_mnist", "caltech101_silhouettes28"]
MODELS = ['rbm', 'orbm', 'irbm']


def build_train_rbm_argparser(subparser):
    DESCRIPTION = "Train an RBM."

    p = subparser.add_parser("rbm", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, choices=DATASETS, metavar="DATASET",
                   help='dataset to train on [{0}].'.format(', '.join(DATASETS))),

    # Model options (RBM)
    model = p.add_argument_group("RBM arguments")
    model.add_argument('size', type=int,
                       help='size of hidden layer.')
    model.add_argument('--cdk', metavar='K', type=int,
                       help='number of Gibbs sampling steps in Contrastive Divergence.', default=1)
    model.add_argument('--PCD', action='store_true', help='use Persistent Contrastive Divergence')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')


def build_train_orbm_argparser(subparser):
    DESCRIPTION = "Train an ordered RBM."

    p = subparser.add_parser("orbm", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, choices=DATASETS, metavar="DATASET",
                   help='dataset to train on [{0}].'.format(', '.join(DATASETS))),

    # Model options (oRBM)
    model = p.add_argument_group("oRBM arguments")
    model.add_argument('size', type=int,
                       help='size of hidden layer.')
    model.add_argument('--cdk', metavar='K', type=int,
                       help='number of Gibbs sampling steps in Contrastive Divergence.', default=1)
    model.add_argument('--PCD', action='store_true', help='use Persistent Contrastive Divergence')
    model.add_argument('--beta', type=float, help='$\\beta$ hyperparameter in penalty term (see paper). Default=1.01', default=1.01)

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')


def build_train_irbm_argparser(subparser):
    DESCRIPTION = "Train an infinite RBM."

    p = subparser.add_parser("irbm", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, choices=DATASETS, metavar="DATASET",
                   help='dataset to train on [{0}].'.format(', '.join(DATASETS))),

    # Model options (iRBM)
    model = p.add_argument_group("iRBM arguments")
    model.add_argument('--size', type=int,
                       help='size of hidden layer. Default 1.', default=1)
    model.add_argument('--cdk', metavar='K', type=int,
                       help='number of Gibbs sampling steps in Contrastive Divergence.', default=1)
    model.add_argument('--PCD', action='store_true', help='use Persistent Contrastive Divergence')
    model.add_argument('--beta', type=float, help='$\\beta$ hyperparameter in penalty term (see paper). Default=1.01', default=1.01)
    model.add_argument('--shrinkable', action='store_true', help='allows the model to shrink using the heuristic mentioned in the paper.')
    model.add_argument('--nb-neurons-to-add', type=int, help='nb of hidden units to add when model is growing. Default: 1', default=1)
    # model.add_argument('--random-init', action='store_true', help='if specified, added hidden units weights will be randomly initialized (hack to help breaking symmetry).')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')


def buildArgsParser():
    DESCRIPTION = ("Script to train an RBM-like model on a dataset"
                   " (binarized MNIST or CalTech101 Silhouettes) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    duration = p.add_argument_group("Duration arguments")
    duration = duration.add_mutually_exclusive_group(required=True)
    duration.add_argument('--nb-epochs', metavar='N', type=int,
                          help='train for N epochs.')
    duration.add_argument('--max-epoch', metavar='N', type=int,
                          help='train for a maximum of N epochs.')

    # Training options
    training = p.add_argument_group("Training arguments")
    training.add_argument('--batch-size', type=int, metavar="M",
                          help='size of the batch to use when training the model. Default: 100.', default=100)
    training.add_argument('--dataset-percent', type=float, metavar="X",
                          help='percent of train data used for training. (Value between 0 and 1)', default=1.)

    # Update rule choices
    update_rules = p.add_argument_group("Update Rules (required)")
    update_rules = update_rules.add_mutually_exclusive_group(required=True)
    update_rules.add_argument('--ConstantLearningRate', metavar="LR", type=str, help='use constant learning rate in training.')
    update_rules.add_argument('--ADAGRAD', metavar="LR [EPS=1e-6]", type=str, help='use ADAGRAD in training.')

    # Regularization choices
    update_rules = p.add_argument_group("Regularization (optional)")
    update_rules = update_rules.add_mutually_exclusive_group(required=False)
    update_rules.add_argument('--L1Regularization', metavar="LAMBDA", type=float, help='use L1 regularization to train model.')
    update_rules.add_argument('--L2Regularization', metavar="LAMBDA", type=float, help='use L2 regularization to train model.')

    # General options (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('--name', type=str,
                         help='name of the experiment. Default: name is generated from arguments.')
    general.add_argument('--seed', type=int,
                         help='seed used to generate random numbers. Default=1234.', default=1234)
    general.add_argument('--keep', type=int, metavar="K",
                         help='if specified, keep a copy of the model each K epoch.')

    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')

    subparser = p.add_subparsers(title="Models", metavar="", dest="model")
    build_train_rbm_argparser(subparser)
    build_train_orbm_argparser(subparser)
    build_train_irbm_argparser(subparser)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Extract experiments hyperparameters
    hyperparams = dict(vars(args))
    # Remove hyperparams that should not be part of the hash
    del hyperparams['nb_epochs']
    del hyperparams['max_epoch']
    del hyperparams['keep']
    del hyperparams['force']
    del hyperparams['name']

    # Get/generate experiment name
    experiment_name = args.name
    if experiment_name is None:
        experiment_name = utils.generate_uid_from_string(repr(hyperparams))

    # Create experiment folder
    experiment_path = pjoin(".", "experiments", experiment_name)
    resuming = False
    if os.path.isdir(experiment_path) and not args.force:
        resuming = True
        print "### Resuming experiment ({0}). ###\n".format(experiment_name)
        # Check if provided hyperparams match those in the experiment folder
        hyperparams_loaded = utils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
        if hyperparams != hyperparams_loaded:
            print "{\n" + "\n".join(["{}: {}".format(k, hyperparams[k]) for k in sorted(hyperparams.keys())]) + "\n}"
            print "{\n" + "\n".join(["{}: {}".format(k, hyperparams_loaded[k]) for k in sorted(hyperparams_loaded.keys())]) + "\n}"
            print "The arguments provided are different than the one saved. Use --force if you are certain.\nQuitting."
            exit(1)
    else:
        if os.path.isdir(experiment_path):
            shutil.rmtree(experiment_path)

        os.makedirs(experiment_path)
        utils.save_dict_to_json_file(pjoin(experiment_path, "hyperparams.json"), hyperparams)

    with Timer("Loading dataset"):
        trainset, validset, testset = dataset.load(args.dataset, args.dataset_percent)
        print " (data: {:,}; {:,}; {:,}) ".format(len(trainset), len(validset), len(testset)),

    with Timer("\nCreating model"):
        model = model_factory(args.model, input_size=trainset.input_size, hyperparams=hyperparams)

    starting_epoch = 1
    if resuming:
        with Timer("\nLoading model"):
            status = utils.load_dict_from_json_file(pjoin(experiment_path, "status.json"))
            starting_epoch = status['no_epoch'] + 1
            model = model.load(pjoin(experiment_path, "model.pkl"))

    ### Build trainer ###
    with Timer("\nBuilding trainer"):
        trainer = Trainer(model, trainset, batch_size=hyperparams['batch_size'], starting_epoch=starting_epoch)

        # Add stopping criteria
        ending_epoch = args.max_epoch if args.max_epoch is not None else starting_epoch + args.nb_epochs - 1
        # Stop when max number of epochs is reached.
        trainer.add_stopping_criterion(tasks.MaxEpochStopping(ending_epoch))

        # Print time a training epoch took
        trainer.add_task(tasks.PrintEpochDuration())
        avg_reconstruction_error = tasks.AverageReconstructionError(model.CD.chain_start, model.CD.chain_end, len(trainset))
        trainer.add_task(tasks.Print(avg_reconstruction_error, msg="Avg. reconstruction error: {0:.1f}"))

        if args.model == 'irbm':
            trainer.add_task(irbm.GrowiRBM(model, shrinkable=args.shrinkable, nb_neurons_to_add=args.nb_neurons_to_add))  #, random_init=args.random_init))

        # Save training progression
        trainer.add_task(tasks.SaveProgression(model, experiment_path, each_epoch=1))
        if args.keep is not None:
            trainer.add_task(tasks.KeepProgression(model, experiment_path, each_epoch=args.keep))

        trainer.build()

    print "\nWill train {0} from epoch {1} to epoch {2}.".format(args.model, starting_epoch, ending_epoch)
    trainer.train()

    with Timer("\nSaving"):
        # Save final model
        model.save(pjoin(experiment_path, "model.pkl"))

if __name__ == "__main__":
    main()
