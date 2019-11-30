import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import abc

from abc import ABC, abstractmethod

from models.mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel



def get_optimizer(model, lr=0.0002, momentum=0.0):
    if optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr)


def get_scheduler(self, optimizer, scheduler):
    if scheduler == 'cosine':
        eta_min = self.eta_min if self.eta_min is not None else 0
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_step, eta_min=eta_min)
    elif scheduler == 'inv_sqrt':
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and self.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > self.warmup_step \
                       else step / (self.warmup_step ** 1.5)
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    elif scheduler == 'dev_perf':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_rate, patience=self.patience, min_lr=self.lr_min)
    elif scheduler == 'constant':
        pass



class SequenceModel(ABC):
    """
    The abstract model class for training.
    """
    def __init__(self, depth, width):
        self.depth = depth
        self.width = width

        self.model = self.init_model(depth, width, self.get_default_hyperparams())


    @abstractmethod
    def init_model(self, depth, width, **hyperparams):
        """
        Returns a model of the specified depth and width.
        """
        ### Each subclass should implement this on their own
        raise NotImplementedError()

    @abstractmethod
    def get_default_hyperparams(self):
        """
        Gets the default hyperparameters for a model.
        Returns:
            A dictioanry of hyperparameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_step(self, batch_X, batch_Y, learning_rate):
        """
        Takes a step of training with a batch of X's and a batch of labels.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_performance(self, X, Y):
        """
        Gets the loss and perplexity for a given X,Y.
        """
        raise NotImplementedError()