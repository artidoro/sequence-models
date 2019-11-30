"""The main class for running experiments.
"""
import logging
import os
import argparse
import json
import config as c
import time

import torch
import torch.optim as optim

from models.sequence_model import get_optimizer, get_scheduler, SequenceModel

from os.path import exists as E
from os.path import join as J

logger = logging.getLogger(__name__)


def set_spec_default_values(spec):
    DEFAULT_VALUES = {
        'device': 'gpu',
        'lr': 0.0002,
        'max_step': 100000,
        'momentum': 0.0,
        'scheduler': 'constant',
        'warmup_step': 0,
    }

    for key, value in DEFAULT_VALUES.items():
        if key not in spec:
            spec[key] = value
    return spec


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


def run_experiment(spec, experiment_directory):
    """Runs an experiment based on the desired experiment specification.
    This process will record the desired response variables and write them to the experiment directory.
    
    Args:
        spec (dict): The JSON object specifying the experiment to run.
        experiment_directory (str):  The directory path to which to write the response variables.
    """
    # spec, experiment_directory = args
    
    # Unpack some of the specification information
    try:
        name = spec["name"]
        # Unpack additional arguments <here>

        spec = set_spec_default_values(spec)

    except KeyError:
        logger.error("Invalid experiment specification: {}".format(spec))
        raise


    # Create the directory
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    else:
        assert c.EXPERIMENT_RUNNER_SHOULD_OVERWRITE, "Experiment directory {} already exists".format(experiment_directory)

    # Output a copy of the experiment specification
    with open(J(experiment_directory, 'params.json'), 'w') as f:
        json.dump(spec, f)
    
    # Todo Run the actual experiment here <> @Ini
    # For now let's just print out the specification

    # TODO: initialize dataset iterators (i.e. `train_iter`)
    # TODO: actually initialize `model`
    sequence_model = SequenceModel(8, 64, {})
    model = sequence_model.model

    # # # # # # #
    # Expects:
    #   model
    #   spec['optimizer'] in ['sgd', 'adam', 'adagrad']
    #   spec['lr'] (float)
    #   Optional: spec['momentum'] (float)
    optimizer = get_optimizer(model, **spec)
    scheduler = get_scheduler(optimizer, **spec)

    max_step = spec['max_step']
    train_step = 0
    train_loss = 0
    best_val_loss = None

    try:
        for epoch in itertools.count(start=1):
            model.train()
            mems = tuple()
            for batch, (data, target, seq_len) in enumerate(train_iter):
                model.zero_grad()

                ret = model.to(spec['device'])(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss)
                if self.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
                self.train_loss += loss.float().item()

                # Gradient clipping
                if self.clip is not None:
                    if self.fp16:
                        self.optimizer.clip_master_grads(self.clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()

                # Step-wise learning rate annealing
                train_step += 1
                scheduler_type = spec['scheduler']
                warmup_step = spec['warmup_step']
                if scheduler_type in ['cosine', 'constant', 'dev_perf']:
                    # linear warmup stage
                    if train_step < warmup_step
                        curr_lr = spec['lr'] * train_step / warmup_step
                        optimizer.param_groups[0]['lr'] = curr_lr
                    else:
                        if scheduler_type.scheduler == 'cosine':
                            scheduler.step(train_step)
                elif self.scheduler_type == 'inv_sqrt':
                    scheduler.step(train_step)

                # TODO: Logging, validation

                if train_step >= max_step: 
                    break

            if train_step >= max_step:
                print('-' * 100)
                print('End of training')

    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')


    # DO SOMETHING WITH THIS SPEC DUDE!
    print(spec)
    # time.sleep(10)
    # THE GPU
    print(os.environ.get(c.CVISIBLE, None))
    
    


    

if __name__ == '__main__':
    # One can also run the experiment directly:
    parser = argparse.ArgumentParser(description="Runs an experiment directly from its JSON.")
    parser.add_argument('specification_json', type=str, help='The JSON which specifies the experiment.')
    parser.add_argument('output_dir', type=str, help='The directory where the output will be written.')
    args = parser.parse_args()

    with open(args.specification_json, 'r') as f:
        spec = json.load(f)

    run_experiment(spec, os.path.join(args.output_dir, spec["name"]))