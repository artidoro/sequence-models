"""Specifies non variable parameters for the experimentation framework.
"""
from collections import OrderedDict

EXPERIMENT_RUNNER_SHOULD_OVERWRITE = True
CVISIBLE = 'CUDA_VISIBLE_DEVICES'

HYPERPARAMETERS = OrderedDict({
    'algorithms': [
        'lstm',
    ],
    'depth': [2],
    'width': [16],
    'vocab': [5000],
    'sequence_dependence': [2]
})