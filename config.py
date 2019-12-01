"""Specifies non variable parameters for the experimentation framework.
"""
from collections import OrderedDict

EXPERIMENT_RUNNER_SHOULD_OVERWRITE = True
CVISIBLE = 'CUDA_VISIBLE_DEVICES'

DATA_GENERATION_VERSION = 3



#   - Vocabulary Size (between 5k and 10k)
#     - Time Dependence (exponential sweep between 2, 4, 8, 16, 32)
#     - Data generation (Zipf's law)
#     - Depth (2 4 8 16) 
#     - Width ()
#     - Choice of optimization = (Adam, SGD later)
#     - Batch size + learn rate decay
#     - Sentence length = (N + memory)

# {"algorithm": "transformer", "depth": 8, "width": 32, "sequence_dependence": 4, "vocab": 1000, "hmm_hidden": 50, "batch_size": 32, "name": "transformer_1", "lr": 0.0001, "multi_gpu": "True", "clip": 0.25,
# "scheduler_type": "constant", "warmup_step": 2000}

HYPERPARAMETERS = OrderedDict({
    'algorithm': [
        'lstm',
        'cnn',
        'transformer'
    ],
    'depth': [2, 4, 8],
    'width': [32, 128, 256, 512],
    'sequence_dependence': [2, 4, 8, 16],
    'vocab': [1000],
    'hmm_hidden': [50],
    'batch_size': [32],
})

ALGORITHM_SPECIFIC_PARAMETERS = {
    'transformer': { 
        "lr": 0.0001,
        "dropout": 0.1, 
        "multi_gpu": "False", 
        "clip": 0.25,
        "scheduler_type": "constant",
        "warmup_step": 2000
    },
    'cnn': {
        "lr": 2e-4,
    },
    'lstm': {
        'lr': 0.001
    }
}

DEFAULT_VALUES_SPEC = {
    'device': 'cuda',
    'max_step': 100000,
    'eval_steps': 10,  # This is similar to the epoch in terms of number of iterations.
    'test_size': 9999, # Number of words in the test set.
    'eta_min': 0.0,
    'embedding_dim': 256,
    'momentum': 0.0,
    'scheduler_type': 'constant',
    'optimizer_type': 'adam',
    'warmup_step': 0,
    'bptt_len': 32
}

REQUIRED_ATTRIBUTES = set(
    list(DEFAULT_VALUES_SPEC.keys()) +
    list(HYPERPARAMETERS.keys())
)
