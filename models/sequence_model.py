import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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



def SequenceModel:

    # # # # # # 
    # Arguments:
    #       depth (int)
    #       width (int)
    #       hyperparams (dict)
    def __init__(self, depth, width, hyperparams):
        # Set hyperparameter values
        for k, v in hyperparams.items():
            setattr(self, k, v)

        self.model = self.init_model(depth, width)


    def init_model(self, depth, width):
        ### Each subclass should implement this on their own





