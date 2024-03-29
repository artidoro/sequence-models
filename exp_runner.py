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
import numpy as np

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from models.gated_cnn import GatedCNN
from models.lstm import LSTMModel
from models.sequence_model import SequenceModel
from models.transformerXL import TransformerXL
from data_generation.data_utils import torchtext_batch_iterators
from data_generation.data_utils import torchtext_batch_iterators_split
from models.utils.tqdm_logger import TqdmLogger

import tqdm

from os.path import exists as E
from os.path import join as J

import config as c


def set_spec_default_values(spec):
    for key, value in c.DEFAULT_VALUES_SPEC.items():
        if key not in spec:
            spec[key] = value
    return spec


def evaluate_model(sequence_model, eval_iter, max_iterations, vocab):
    """
    Computes perplexity of a given model on an evaluation iterator.
    """
    # Transformer has its own evaluation method
    if isinstance(sequence_model, TransformerXL):
        return sequence_model.evaluate(eval_iter, max_iterations)

    cross_entropy_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        cross_entropy_loss = nn.CrossEntropyLoss()

        total_cross_entropy = 0
        num_batches = 0
        acc = []

        for idx, batch in tqdm.tqdm(enumerate(eval_iter)):
            predictions = sequence_model.predict(batch.text)
            percentage_correct = np.mean(
                np.argmax(predictions.detach().cpu().numpy(), axis=-1) ==  batch.target.cpu().numpy()
            )

            acc.append(percentage_correct)

            cross_ent = cross_entropy_loss(predictions.reshape(-1, vocab), batch.target.flatten())
            total_cross_entropy += cross_ent
            num_batches += 1

            if idx >= max_iterations:
                break

    if (num_batches == 0):
        return float('inf'), 0

    return math.exp(total_cross_entropy / num_batches), np.mean(acc)


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
        spec = set_spec_default_values(spec)

        algorithm = spec["algorithm"]
        batch_size = spec['batch_size']
        bptt_len = spec['bptt_len']
        device = spec['device']
        hmm_hidden = spec['hmm_hidden']
        max_step = spec['max_step']
        name = spec['name']
        sequence_dependence = spec['sequence_dependence']
        vocab = spec['vocab']
        # Unpack additional arguments <here>

    except KeyError:
        print("Invalid experiment specification: {}".format(spec))
        raise

    logging.basicConfig(level=logging.DEBUG)
                    # filename=J(experiment_directory, 'out.log'),
                    # filemode='w')
    logger = logging.getLogger('exp_runner')


    logger.info("Starting the experiment!")
    logger.info(str(spec))

    # Create the directory
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    else:
        assert c.EXPERIMENT_RUNNER_SHOULD_OVERWRITE, "Experiment directory {} already exists".format(experiment_directory)

    # Output a copy of the experiment specification
    with open(J(experiment_directory, 'params.json'), 'w') as f:
        json.dump(spec, f)

    # Choose sequence model type
    if algorithm == 'transformer':
        sequence_model = TransformerXL(**spec)
    elif algorithm == 'lstm':
        sequence_model = LSTMModel(**spec)
    elif algorithm == 'cnn':
        sequence_model = GatedCNN(**spec)
    else:
        print(spec)

    # TODO: loop over trainig files/algorithm specification
    ROOT_PATH = 'generated_data'
    DATA_FILE = 'V{}_hmm_hidden_{}_lag_{}_vocab_{}.txt'.format(
        c.DATA_GENERATION_VERSION, hmm_hidden, sequence_dependence, vocab)
    train_file = 'train_' + DATA_FILE
    test_file = 'test_' + DATA_FILE
    device = torch.device(device)

    # Create dataset iterators
    train_iter, test_iter = torchtext_batch_iterators(
        ROOT_PATH, train_file, test_file,
        batch_size=batch_size, bptt_len=bptt_len, device=device, batch_first=True, repeat=False)

    train_perplex_iter, test_perplex_iter = torchtext_batch_iterators(
        ROOT_PATH, train_file, test_file,
        batch_size=batch_size, bptt_len=bptt_len, device=device, batch_first=True, repeat=False)

    # Model
    model = sequence_model.get_model()
    optimizer = sequence_model.get_optimizer()
    scheduler = sequence_model.get_scheduler()

    max_step = spec['max_step']
    eval_steps = spec["eval_steps"]
    train_step = 0
    train_loss = 0
    best_val_loss = None

    losses = []
    test_performance = []
    train_performance = []
    step_to_performance = []

    num_steps = 0
    # Training Loop

    tqdm_out = TqdmLogger(logger,level=logging.INFO)
    progress = tqdm.tqdm(total=max_step,)

    try:
        for epoch in itertools.count(start=1):
            model.train()
            mems = tuple()
            print()

            for train_step, batch in enumerate(train_iter):
                num_steps +=1
                progress.update()
                loss = sequence_model.train_step(batch.text, batch.target, mems=mems)
                losses.append(loss)
                progress.set_description("Loss {:.4f}".format(loss))

                # Update scheduler
                sequence_model.update_scheduler(num_steps)


                if num_steps % 500 == 0:
                    progress.write("Saving loss performance!")
                    np.save(J(experiment_directory, 'losses.npy'), losses)
                    np.save(J(experiment_directory, 'test_performance.npy'), test_performance)
                    np.save(J(experiment_directory, 'train_performance.npy'), train_performance)
                    np.save(J(experiment_directory, 'step_to_performance.npy'), step_to_performance)
                
                if num_steps % 1000 == 0:
                    # Calculate perplexity
                    progress.write("-"* 100)
                    progress.write("Model Performance:")
                    test_performance.append(evaluate_model(sequence_model, test_perplex_iter, 2000, vocab))
                    train_performance.append(evaluate_model(sequence_model, train_perplex_iter, 1000, vocab))
                    step_to_performance.append(num_steps)
                    progress.write("Test (Perplex, Accuracy): {:.6f}, {:.6f}".format(*test_performance[-1]))
                    progress.write("Train (Perplex, Accuracy): {:.6f}, {:.6f}".format(*train_performance[-1]))
                    progress.write("Average loss (past 1000): {}".format(np.mean(losses[-1000:])))
                    
                if num_steps >= max_step:
                    break

        

            if num_steps >= max_step:
                progress.write('-' * 100)
                progress.write('End of training')
                break

            # if val_loss is None or val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     # TODO: save the best performing model so far(and its stats)

    except KeyboardInterrupt:
        logger.info('-' * 100)
        logger.info('Exiting from training early')
        raise



def parameter_count(spec, experiment_directory):
    # spec, experiment_directory = args
    
    # Unpack some of the specification information
    try:
        spec = set_spec_default_values(spec)

        algorithm = spec["algorithm"]
        batch_size = spec['batch_size']
        bptt_len = spec['bptt_len']
        spec['device'] = 'cpu'
        device = 'cpu'
        hmm_hidden = spec['hmm_hidden']
        max_step = spec['max_step']
        name = spec['name']
        sequence_dependence = spec['sequence_dependence']
        vocab = spec['vocab']
        # Unpack additional arguments <here>

    except KeyError:
        print("Invalid experiment specification: {}".format(spec))
        raise

    logging.basicConfig(level=logging.DEBUG)
                    # filename=J(experiment_directory, 'out.log'),
                    # filemode='w')
    logger = logging.getLogger('exp_runner')


    logger.info("Starting the parameter counter!")
    logger.info(str(spec))

    # Create the directory
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    else:
        assert c.EXPERIMENT_RUNNER_SHOULD_OVERWRITE, "Experiment directory {} already exists".format(experiment_directory)

    # Choose sequence model type
    if algorithm == 'transformer':
        sequence_model = TransformerXL(**spec)
    elif algorithm == 'lstm':
        sequence_model = LSTMModel(**spec)
    elif algorithm == 'cnn':
        sequence_model = GatedCNN(**spec)
    else:
        print(spec)

    # Model
    model = sequence_model.get_model()
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print(pp)
    np.save(J(experiment_directory, 'parameters.npy'), [pp])


if __name__ == '__main__':
    # One can also run the experiment directly:
    parser = argparse.ArgumentParser(description="Runs an experiment directly from its JSON.")
    parser.add_argument('specification_json', type=str, help='The JSON which specifies the experiment.')
    parser.add_argument('output_dir', type=str, help='The directory where the output will be written.')
    args = parser.parse_args()

    with open(args.specification_json, 'r') as f:
        spec = json.load(f)

    run_experiment(spec, os.path.join(args.output_dir, spec["name"]))

    