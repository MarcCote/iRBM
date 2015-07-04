import numpy as np

from iRBM.misc.learning_rate import (ConstantLearningRate,
                                     ADAGRAD)

from iRBM.misc.regularization import (NoRegularization,
                                      L1Regularization,
                                      L2Regularization)

from iRBM.misc.contrastive_divergence import (ContrastiveDivergence,
                                              PersistentCD)


def model_factory(model_name, input_size, hyperparams):
    #Set learning rate method that will be used.
    if hyperparams["ConstantLearningRate"] is not None:
        infos = hyperparams["ConstantLearningRate"].split()
        lr = float(infos[0])
        lr_method = ConstantLearningRate(lr=lr)
    elif hyperparams["ADAGRAD"] is not None:
        infos = hyperparams["ADAGRAD"].split()
        lr = float(infos[0])
        eps = float(infos[1]) if len(infos) > 1 else 1e-6
        lr_method = ADAGRAD(lr=lr, eps=eps)
    else:
        raise ValueError("The update rule is mandatory!")

    #Set regularization method that will be used.
    regularization_method = NoRegularization()
    if hyperparams["L1Regularization"] is not None:
        lambda_factor = float(hyperparams["L1Regularization"])
        regularization_method = L1Regularization(lambda_factor)
    elif hyperparams["L2Regularization"] is not None:
        lambda_factor = float(hyperparams["L1Regularization"])
        regularization_method = L2Regularization(lambda_factor)

    #Set contrastive divergence method to use.
    CD_method = ContrastiveDivergence()
    if hyperparams["PCD"]:
        CD_method = PersistentCD(input_size, nb_particles=hyperparams['batch_size'])

    rng = np.random.RandomState(hyperparams["seed"])

    #Build model
    if model_name == "rbm":
        from iRBM.models.rbm import RBM
        model = RBM(input_size=input_size,
                    hidden_size=hyperparams["size"],
                    learning_rate=lr_method,
                    regularization=regularization_method,
                    CD=CD_method,
                    CDk=hyperparams["cdk"],
                    rng=rng
                    )

    elif model_name == "orbm":
        from iRBM.models.orbm import oRBM
        model = oRBM(input_size=input_size,
                     hidden_size=hyperparams["size"],
                     beta=hyperparams["beta"],
                     learning_rate=lr_method,
                     regularization=regularization_method,
                     CD=CD_method,
                     CDk=hyperparams["cdk"],
                     rng=rng
                     )

    elif model_name == "irbm":
        from iRBM.models.irbm import iRBM
        model = iRBM(input_size=input_size,
                     hidden_size=hyperparams["size"],
                     beta=hyperparams["beta"],
                     learning_rate=lr_method,
                     regularization=regularization_method,
                     CD=CD_method,
                     CDk=hyperparams["cdk"],
                     rng=rng
                     )

    return model
