"""The main class for running experiments.
"""
import logging
import os
import argparse
import json
import config as c
import time
import itertools
import math

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from models.gated_cnn import GatedCNN
from models.lstm import LSTMModel
from models.sequence_model import SequenceModel
from models.transformerXL import TransformerXL
from data_generation.data_utils import torchtext_batch_iterators

from os.path import exists as E
from os.path import join as J

import config as c

logger = logging.getLogger(__name__)


def set_spec_default_values(spec):
    for key, value in c.DEFAULT_VALUES_SPEC.items():
        if key not in spec:
            spec[key] = value
    return spec

def evaluate_model(sequence_model, eval_iter, max_iterations):
    cross_entropy_loss = nn.CrossEntropyLoss()
    total_perplexity = 0

    for idx, batch in tqdm(enumerate(eval_iter)):
        predictions = sequence_model.predict(batch.text)
        cross_ent = cross_entropy_loss(predictions.view(-1, predictions.shape[-1]), batch.target.flatten())
        perp = math.exp(cross_ent.item())
        total_perplexity += perp

        if idx >= max_iterations:
            break

    return total_perplexity / max_iterations

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
        hmm_hidden = spec["hmm_hidden"]
        vocab = spec["vocab"]
        batch_size = spec["batch_size"]
        sequence_dependence = spec["sequence_dependence"]
        bttp_len = spec["bttp_len"]
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

    if spec["algorithm"] == 'transformer':
        sequence_model = TransformerXL(**spec)
    elif spec["algorithm"] == 'lstm':
        sequence_model = LSTMModel(**spec)
    elif spec["algorithm"] == 'cnn':
        sequence_model = GatedCNN(**spec)

    data_file = 'V{}hmm_hidden_{}_lag_{}_vocab_{}.txt'.format(
        c.DATA_GENERATION_VERSION, hmm_hidden, sequence_dependence, vocab)

    model = sequence_model.get_model()
    optimizer = sequence_model.get_optimizer()
    scheduler = sequence_model.get_scheduler()

    max_step = spec['max_step']
    eval_steps = spec["eval_steps"]
    train_step = 0
    train_loss = 0
    best_val_loss = None

    # Write training universal training code for every mode.
    train_iter, validation_iter, test_iter = torchtext_batch_iterators('generated_data',
        'train/' + data_file, 'validation/' + data_file, 'test/' + data_file,
        batch_size=batch_size, bptt_len=bttp_len, device=None, batch_first=False, repeat=True)
        # TODO: pass the device to this call so that the data is already on the GPU.

    # Training Loop
    try:
        # TODO: somewhere in here compute perplexity or validation loss or whatever, and save the best performing models with their corresponding stats
        for epoch in itertools.count(start=1):
            model.train()
            mems = tuple()
            for train_step, batch in enumerate(train_iter):
                #1. Get batch of paragraphs/documents  (batch, seq_len)
                sequence_model.train_step(batch.text, batch.target)
                # TODO: unify the problem of train_step and mems as argument in Transformer XL and the other models.

                #2. Train
                loss = sequence_model.train_step(batch.text, batch.target)
                print(loss)

                #3. Compute perplexity
                if train_step % eval_steps == eval_steps - 1:
                    avg_perp = evaluate_model(sequence_model, test_iter, eval_size)
                    print(avg_perp)


                #4. Update the scheduler.
                # repeat.

                if train_step >= max_step:
                    break

            if train_step >= max_step:
                print('-' * 100)
                print('End of training')

    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')


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

    