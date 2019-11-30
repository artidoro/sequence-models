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


DEFAULT_VALUES_SPEC = {
    'device': 'cuda',
    'lr': 0.0002,
    'max_step': 100000,
    'momentum': 0.0,
    'scheduler': 'constant',
    'warmup_step': 0,
}

REQUIRED_ATTRIBUTES = set(
    list(DEFAULT_VALUES_SPEC.keys()) +
    list(HYPERPARAMETERS.keys.keys())
)