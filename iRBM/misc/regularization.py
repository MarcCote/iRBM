REGULARIZATION_METHODS = ["no", "L1", "L2"]


class Regularization():
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, param):
        raise NameError('Should be implemented by subclasses!')


class NoRegularization(Regularization):
    def __init__(self):
        Regularization.__init__(self, 0.0)

    def __call__(self, param):
        return 0.0


class L1Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        return self.decay * abs(param).sum()


class L2Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        return 2*self.decay * (param**2).sum()
