import torch
import torch.optim as optim
import abc

from abc import ABC, abstractmethod

import config as c

class SequenceModel(ABC):
    """
    The abstract model class for training.
    """
    def __init__(self, **hparams):
        for k in c.REQUIRED_ATTRIBUTES:
            assert k in hparams, "Missing required attribute {}".format(k)

        for key, value in hparams.items():
            setattr(self, key, value)

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

    @abstractmethod
    def init_model(self):
        """
        Returns a model of the specified depth and width.
        """
        ### Each subclass should implement this on their own
        raise NotImplementedError()


    def init_optimizer(self):
        """Initializes the optimizer
        """
        if self.optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type.lower() == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.lr)


    def init_scheduler(self):
        """Initializes the scheduler
        """
        if self.scheduler_type == 'cosine':
            eta_min = self.eta_min if self.eta_min is not None else 0
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_step, eta_min=eta_min)
        elif self.scheduler_type == 'inv_sqrt':
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and self.warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > self.warmup_step \
                        else step / (self.warmup_step ** 1.5)
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.scheduler_type == 'dev_perf':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.decay_rate, patience=self.patience, min_lr=self.lr_min)


    @abstractmethod
    def predict(self, inputs):
        """
        Gets predictions for the next token of a batch of sequences (as a distribution over vocab tokens).
        
        Arguments:
            inputs : a Tensor of shape (input_seq_length, batch_size)

        Returns:
            probs : a Tensor of shape (batch_size, vocab_size)
        """
        raise NotImplementedError()

    
    @abstractmethod
    def train_step(self, inputs, targets, train_step=0, mems=tuple()):
        """
        Performs an unsupervised train step for a given batch.
        Returns loss on batch.

        `train_step` is needed whenever the scheduler is non-constant
        `mems` is only used for TransformerXL
        """
        raise NotImplementedError()

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler
