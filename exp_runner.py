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

from models.gated_cnn import GatedCNN
from models.lstm import LSTMModel
from models.sequence_model import SequenceModel
from models.transformerXL import TransformerXL

from os.path import exists as E
from os.path import join as J

import config as c

logger = logging.getLogger(__name__)


def set_spec_default_values(spec):
    for key, value in c.DEFAULT_VALUES_SPEC.items():
        if key not in spec:
            spec[key] = value
    return spec



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
    train_iter = [] ### REPLACE 
    val_iter = [] ##### WITH
    test_iter = [] #### ACTUAL ITERATORS

    if spec["algorithm"] == 'transformer':
        sequence_model = TransformerXL(**spec)
    elif spec["algorithm"] == 'lstm':
        sequence_model = LSTMModel(**spec)
    elif spec["algorithm"] == 'cnn':
        sequence_model = GatedCNN(**spec)

    model = sequence_model.get_model()
    optimizer = sequence_model.get_optimizer()
    scheduler = sequence_model.get_scheduler()

    max_step = spec['max_step']
    train_step = 0
    train_loss = 0
    best_val_loss = None


    # Training Loop
    try:
        # TODO: somewhere in here compute perplexity or validation loss or whatever, and save the best performing models with their corresponding stats
        for epoch in itertools.count(start=1):
            model.train()
            mems = tuple()
            for batch_inputs, batch_targets in enumerate(train_iter):
                sequence_model.train_step(batch_inputs, batch_targets, train_step=train_step, mems=mems)

                if train_step >= max_step: 
                    break

            if train_step >= max_step:
                print('-' * 100)
                print('End of training')

    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')


    # Write training universal training code for every mode.
    # Run for some number of training iterations.
    #1. Get batch of paragraphs/documents  (batch, seq_len)
    # TODO
    #2. Train 
    loss = sequence_model.train_step(batch_inputs, batch_targets)
    #3. Compute perplexity
    predictions = sequence_model.predict(batch_inputs)
    # TODO: use this for perplexity
    #4.s Update the scheduler.
    # repeat.

    
    ############################################################
    ### Example train_step() implementation <from transformerXL>
    ############################################################
    # def train_step(self, inputs, targets, mems=tuple(), train_step=0):
    #     """
    #     Performs an unsupervised train step for a given batch.
    #     Returns loss on batch.
    #     """

    #     # Zero out model gradients
    #     self.model.zero_grad()

    #     # Calculate loss
    #     ret = self.para_model(inputs, targets, *mems)
    #     loss, mems = ret[0], ret[1:]
    #     loss = loss.float().mean().type_as(loss)
    #     if self.fp16:
    #         self.optimizer.backward(loss)
    #     else:
    #         loss.backward()

    #     # Gradient clipping
    #     if self.clip is not None:
    #         if self.fp16:
    #             self.optimizer.clip_master_grads(self.clip)
    #         else:
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

    #     self.optimizer.step()

    #     # Update scheduler
    #     self.update_scheduler(train_step)

    #     return loss

    


    

if __name__ == '__main__':
    # One can also run the experiment directly:
    parser = argparse.ArgumentParser(description="Runs an experiment directly from its JSON.")
    parser.add_argument('specification_json', type=str, help='The JSON which specifies the experiment.')
    parser.add_argument('output_dir', type=str, help='The directory where the output will be written.')
    args = parser.parse_args()

    with open(args.specification_json, 'r') as f:
        spec = json.load(f)

    run_experiment(spec, os.path.join(args.output_dir, spec["name"]))

    