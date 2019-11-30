"""Specifies non variable parameters for the experimentation framework.
"""
from collections import OrderedDict

EXPERIMENT_RUNNER_SHOULD_OVERWRITE = True
CVISIBLE = 'CUDA_VISIBLE_DEVICES'

HYPERPARAMETERS = OrderedDict({
    'algorithm': [
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
    'eta_min': 0.0,
    'embedding_dim': 256,
    'momentum': 0.0,
    'scheduler_type': 'constant',
    'optimizer_type': 'sgd',
    'warmup_step': 0,
    'bptt_len': 28
}

REQUIRED_ATTRIBUTES = set(
    list(DEFAULT_VALUES_SPEC.keys()) +
    list(HYPERPARAMETERS.keys())
)
