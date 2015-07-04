# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from time import time
from os.path import join as pjoin

from iRBM.misc import utils


class StoppingCriterion:
    def check(self, no_epoch):
        raise NotImplementedError("Subclass has to implement this function.")


class Task(object):
    def __init__(self):
        self.updates = OrderedDict()

    def init(self, no_epoch, no_update):
        pass

    def pre_epoch(self, no_epoch, no_update):
        pass

    def pre_update(self, no_epoch, no_update):
        pass

    def post_update(self, no_epoch, no_update):
        pass

    def post_epoch(self, no_epoch, no_update):
        pass

    def finished(self, no_epoch, no_update):
        pass


class View(Task):
    def __init__(self):
        super(View, self).__init__()
        self.value = None
        self.last_epoch = -1
        self.last_update = -1

    def view(self, no_epoch, no_update):
        if self.last_epoch != no_epoch or self.last_update != no_update:
            self.update(no_epoch, no_update)
            self.last_epoch = no_epoch
            self.last_update = no_update

        return self.value

    def update(self, no_epoch, no_update):
        raise NotImplementedError("Subclass has to implement this function.")

    def __str__(self):
        return "{0}".format(self.value)


class Print(Task):
    def __init__(self, view, msg="{0}", each_epoch=1, each_update=0):
        super(Print, self).__init__()
        self.msg = msg
        self.each_epoch = each_epoch
        self.each_update = each_update
        self.view_obj = view

        # Get updates of the view object.
        self.updates.update(view.updates)

    def post_update(self, no_epoch, no_update):
        self.view_obj.post_update(no_epoch, no_update)

        if self.each_update != 0 and no_update % self.each_update == 0:
            value = self.view_obj.view(no_epoch, no_update)
            print self.msg.format(value)

    def post_epoch(self, no_epoch, no_update):
        self.view_obj.post_epoch(no_epoch, no_update)

        if self.each_epoch != 0 and no_epoch % self.each_epoch == 0:
            value = self.view_obj.view(no_epoch, no_update)
            print self.msg.format(value)

    def init(self, no_epoch, no_update):
        self.view_obj.init(no_epoch, no_update)

    def pre_epoch(self, no_epoch, no_update):
        self.view_obj.pre_epoch(no_epoch, no_update)

    def pre_update(self, no_epoch, no_update):
        self.view_obj.pre_update(no_epoch, no_update)

    def finished(self, no_epoch, no_update):
        self.view_obj.finished(no_epoch, no_update)


class PrintEpochDuration(Task):
    def __init__(self):
        super(PrintEpochDuration, self).__init__()

    def init(self, no_epoch, no_update):
        self.training_start_time = time()

    def pre_epoch(self, no_epoch, no_update):
        self.epoch_start_time = time()

    def post_epoch(self, no_epoch, no_update):
        print "Epoch {0} done in {1:.03f} sec.".format(no_epoch, time() - self.epoch_start_time)

    def finished(self, no_epoch, no_update):
        print "Training done in {:.03f} sec.".format(time() - self.training_start_time)


class AverageReconstructionError(View):
    def __init__(self, var1, var2, N):
        super(AverageReconstructionError, self).__init__()
        self.N = N
        self.sum_reconstuction_error = theano.shared(np.array(0., dtype=theano.config.floatX))

        # Will be performed after each update
        self.updates[self.sum_reconstuction_error] = self.sum_reconstuction_error + T.sum((var1-var2)**2)

    def pre_epoch(self, no_epoch, no_update):
        self.sum_reconstuction_error.set_value(np.array(0., dtype=theano.config.floatX))

    def update(self, no_epoch, no_update):
        self.value = self.sum_reconstuction_error.get_value() / self.N


class MaxEpochStopping(StoppingCriterion):
    def __init__(self, nb_epochs_max):
        self.nb_epochs_max = nb_epochs_max

    def check(self, no_epoch):
        return no_epoch > self.nb_epochs_max


class ItemGetter(View):
    def __init__(self, view_obj, attribute):
        """ Retrieves `attribute` from a `view` which outputs a dictionnary """
        super(ItemGetter, self).__init__()
        self.view_obj = view_obj
        self.attribute = attribute

    def update(self, no_epoch, no_update):
        infos = self.view_obj.view(no_epoch, no_update)
        self.value = infos[self.attribute]


class SaveProgression(Task):
    def __init__(self, model, savedir, each_epoch=1):
        super(SaveProgression, self).__init__()

        self.savedir = savedir
        self.model = model
        self.each_epoch = each_epoch

    def execute(self, no_epoch, no_update):
        self.model.save(pjoin(self.savedir, "model.pkl"))
        status = {'no_epoch': no_epoch,
                  'no_update': no_update}
        utils.save_dict_to_json_file(pjoin(self.savedir, "status.json"), status)

    def post_epoch(self, no_epoch, no_update):
        if no_epoch % self.each_epoch == 0:
            self.execute(no_epoch, no_update)
