from __future__ import division

import theano
import theano.tensor as T
import numpy as np

from itertools import count


class Trainer(object):
    def __init__(self, model, dataset, batch_size=None, starting_epoch=1):
        self.learn = None
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size if batch_size is not None else len(dataset)
        self.nb_updates = int(np.ceil(len(dataset) / self.batch_size))
        self.starting_epoch = starting_epoch

        self.stopping_criteria = []
        self.tasks = []

        self.no_epoch = 0
        self.no_update = 0
        self.final_epoch = None

        # Build learner
        self.input = T.matrix('input')
        self.no_batch = T.iscalar('no_batch')
        self.updates = self.model.get_updates(self.input)

    def build(self):
        self.learn = theano.function([self.no_batch],
                                     updates=self.updates,
                                     givens={self.input: self.dataset.inputs[self.no_batch * self.batch_size:(self.no_batch + 1) * self.batch_size]},
                                     name="learn")
        #theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(model.__class__.__name__, config.device), with_ids=True)

    def train(self):
        if self.learn is None:
            self.build()

        self.init()

        # Learning
        for self.no_epoch in count(self.starting_epoch):
            # Check stopping criteria
            if any([stopping_criterion.check(self.no_epoch) for stopping_criterion in self.stopping_criteria]):
                break

            self.pre_epoch()

            for self.no_update in xrange(1, self.nb_updates+1):
                self.pre_update()
                self.learn(self.no_update-1)
                self.post_update()

            self.post_epoch()

        self.final_epoch = self.no_epoch-1
        self.finished()

    def add_stopping_criterion(self, criterion):
        self.stopping_criteria.append(criterion)

    def add_task(self, task, epoch_frequency=1, update_frequency=None):
        self.updates.update(task.updates)
        self.tasks.append(task)

    def track_variable(self, var, shape, name=""):
        var_shared = theano.shared(np.zeros(shape, dtype=theano.config.floatX), name=name)
        self.updates[var_shared] = var
        return var_shared

    def init(self):
        for task in self.tasks:
            task.init(self.no_epoch, 0)

    def pre_epoch(self):
        for task in self.tasks:
            task.pre_epoch(self.no_epoch, 0)

    def pre_update(self):
        for task in self.tasks:
            task.pre_update(self.no_epoch, self.no_update)

    def post_update(self):
        for task in self.tasks:
            task.post_update(self.no_epoch, self.no_update)

    def post_epoch(self):
        for task in self.tasks:
            task.post_epoch(self.no_epoch, self.no_update)

    def finished(self):
        for task in self.tasks:
            task.finished(self.final_epoch, self.no_update)
