import torch
import torch.optim as optim



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



class SequenceModel:
    """
    A generic, abstract class representing a sequence model
    """

    def __init__(self, depth, width, hyperparams):
        """
        Arguments:
            depth (int)
            width (int)
            hyperparams (dict) <these become class variables, which can be accessed like `self.param`>
        """

        # Set hyperparameter values
        for k, v in hyperparams.items():
            setattr(self, k, v)

        self.model = self.init_model(depth, width)


    def init_model(self, depth, width):
        pass # Each subclass should implement this on their own


    def get_model(self):
        return self.model


