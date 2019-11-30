"""Specifies non variable parameters for the experimentation framework.
"""
from collections import OrderedDict

EXPERIMENT_RUNNER_SHOULD_OVERWRITE = True
CVISIBLE = 'CUDA_VISIBLE_DEVICES'

HYPERPARAMETERS = OrderedDict({
    'algorithms': [
        'cnn',
        'lstm',
        'lstm_attention',
    ],
    'depth': [2,4,8,16],
    'width': [16,32,64,128,512,1024],
    'vocab': [5000],
    'sequence_dependence': [2,4,8,16,32]
})