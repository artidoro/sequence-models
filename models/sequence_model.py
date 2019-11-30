import torch
import torch.optim as optim
import abc

from abc import ABC, abstractmethod

import config as c



def get_optimizer(model, lr=c.DEFAULT_LEARNING_RATE, momentum=0.0):
    if optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr)


def get_scheduler(self, optimizer, scheduler, eta_min=0, 
    max_step=100000, warmup_step=0, decay_rate=0.5, 
    patience=0, min_lr=0.0):
    
    if scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, eta_min=eta_min)
    elif scheduler == 'inv_sqrt':
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > warmup_step \
                       else step / (warmup_step ** 1.5)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler == 'dev_perf':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=decay_rate, patience=patience, min_lr=lr_min)
    elif scheduler == 'constant':
        pass



class SequenceModel(ABC):
    """
    The abstract model class for training.
    """
    def __init__(self, **hparams):
        for k in c.REQUIRED_ATTRIBUTES:
            assert k in hparams, "Missing required attribute {}".format(k)

        for key, value in hparams.items():
            setattr(self, key, value)

            
        self.model = self.init_model(**hparams)


    @abstractmethod
    def init_model(self, **hparams):
        """
        Returns a model of the specified depth and width.
        """
        ### Each subclass should implement this on their own
        raise NotImplementedError()


    @abstractmethod
    def get_performance(self, X, Y):
        """
        Gets the loss and perplexity for a given X,Y.
        """
        raise NotImplementedError()


    def get_model(self):
        return self.model

