import os
import numpy as np
import theano

from os.path import join as pjoin

from urllib import urlretrieve  # If Python 2.7
#from urllib.request import urlretrieve  # If Python 3

DATASETS_FOLDER = "./datasets"


class Dataset(object):
    def __init__(self, inputs, targets=None, name="dataset"):
        self.name = name
        self.inputs = inputs
        self.targets = targets
        self.symb_inputs = theano.tensor.matrix(name=self.name+'_inputs')
        self.symb_targets = theano.tensor.matrix(name=self.name+'_targets')

    @property
    def inputs(self):
        return self._inputs_shared

    @inputs.setter
    def inputs(self, value):
        self._inputs_shared = theano.shared(value, name=self.name + "_inputs", borrow=True)

    @property
    def targets(self):
        return self._targets_shared

    @targets.setter
    def targets(self, value):
        if value is not None:
            self._targets_shared = theano.shared(value, name=self.name + "_targets", borrow=True)
        else:
            self._targets_shared = None

    @property
    def input_size(self):
        return len(self.inputs.get_value()[0])

    @property
    def target_size(self):
        if self.targets is None:
            return 0
        else:
            return len(self.targets.get_value()[0])

    def __len__(self):
        return len(self.inputs.get_value())


def load_binarized_mnist(percent=1.):
    dataset_name = "binarized_mnist"
    if not os.path.isdir(DATASETS_FOLDER):
        os.mkdir(DATASETS_FOLDER)

    repo = pjoin(DATASETS_FOLDER, dataset_name)
    dataset_npy = pjoin(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.makedirs(repo)

        if not os.path.isfile(pjoin(repo, 'mnist_test.txt')):
            urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt', pjoin(repo, 'mnist_train.txt'))
            urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt', pjoin(repo, 'mnist_valid.txt'))
            urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt', pjoin(repo, 'mnist_test.txt'))

        train_file, valid_file, test_file = [pjoin(repo, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]
        rng = np.random.RandomState(42)

        def parse_file(filename):
            data = np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])
            data = data[:, :-1]  # Remove target
            data = (data > rng.rand(*data.shape)).astype('int8')
            return data

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        np.savez(dataset_npy,
                 trainset_inputs=trainset,
                 validset_inputs=validset,
                 testset_inputs=testset)

    data = np.load(dataset_npy)
    nb_train_data = int(percent*len(data['trainset_inputs']))
    nb_valid_data = int(percent*len(data['validset_inputs']))
    trainset = Dataset(data['trainset_inputs'][:nb_train_data].astype(theano.config.floatX), name="trainset")
    validset = Dataset(data['validset_inputs'][:nb_valid_data].astype(theano.config.floatX), name="validset")
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), name="testset")

    return trainset, validset, testset


def load_caltech101_silhouettes28(percent=1.):
    dataset_name = "caltech101_silhouettes28"
    if not os.path.isdir(DATASETS_FOLDER):
        os.mkdir(DATASETS_FOLDER)

    repo = pjoin(DATASETS_FOLDER, dataset_name)
    dataset_npy = pjoin(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.makedirs(repo)

        if not os.path.isfile(pjoin(repo, 'caltech101_silhouettes28.mat')):
            urlretrieve('http://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat', pjoin(repo, 'caltech101_silhouettes28.mat'))

        import scipy.io
        matlab_file = scipy.io.loadmat(pjoin(repo, 'caltech101_silhouettes28.mat'))

        trainset_inputs = matlab_file['train_data']
        validset_inputs = matlab_file['val_data']
        testset_inputs = matlab_file['test_data']

        np.savez(dataset_npy,
                 trainset_inputs=trainset_inputs,
                 validset_inputs=validset_inputs,
                 testset_inputs=testset_inputs)

    data = np.load(dataset_npy)
    nb_train_data = int(percent*len(data['trainset_inputs']))
    nb_valid_data = int(percent*len(data['validset_inputs']))
    trainset = Dataset(data['trainset_inputs'][:nb_train_data].astype(theano.config.floatX), name="trainset")
    validset = Dataset(data['validset_inputs'][:nb_valid_data].astype(theano.config.floatX), name="validset")
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), name="testset")

    return trainset, validset, testset


def load(dataset_name, percent=1.):
    if dataset_name.lower() == "binarized_mnist":
        return load_binarized_mnist(percent)
    elif dataset_name.lower() == "caltech101_silhouettes28":
        return load_caltech101_silhouettes28(percent)
    else:
        raise ValueError("Unknown dataset: {0}!".format(dataset_name))
