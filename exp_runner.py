"""The main class for running experiments.
"""
import logging
import os
import argparse
import json
import config as c
import time
import itertools

import torch
import torch.optim as optim

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




    # TODO: loop over trainig files/algorithm specification
    data_file = 'V{}hmm_hidden_{}_lag_{}_vocab_{}.txt'.format(
        c.DATA_GENERATION_VERSION, hmm_hidden, sequence_dependence, vocab)

    model = sequence_model.get_model()
    optimizer = sequence_model.get_optimizer()
    scheduler = sequence_model.get_scheduler()

    max_step = spec['max_step']
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
                loss = sequence_model.train_step(batch.text, batch.target, train_step=train_step, mems=mems)

                if train_step >= max_step:
                    break

            if train_step >= max_step:
                print('-' * 100)
                print('End of training')

            # TODO: calculate validation loss & perplexity
            val_loss = None
            perplexity = None

            for val_batch in val_iter:
                preds = sequence_model.predict(val_batch.text)
            
            if val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                # TODO: save the best performing model so far(and its stats)

    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')



if __name__ == '__main__':
    # One can also run the experiment directly:
    parser = argparse.ArgumentParser(description="Runs an experiment directly from its JSON.")
    parser.add_argument('specification_json', type=str, help='The JSON which specifies the experiment.')
    parser.add_argument('output_dir', type=str, help='The directory where the output will be written.')
    args = parser.parse_args()

    with open(args.specification_json, 'r') as f:
        spec = json.load(f)

    run_experiment(spec, os.path.join(args.output_dir, spec["name"]))

    