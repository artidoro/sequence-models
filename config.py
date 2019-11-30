"""Specifies non variable parameters for the experimentation framework.
"""
from collections import OrderedDict

EXPERIMENT_RUNNER_SHOULD_OVERWRITE = True
CVISIBLE = 'CUDA_VISIBLE_DEVICES'

DATA_GENERATION_VERSION = 0



#   - Vocabulary Size (between 5k and 10k)
#     - Time Dependence (exponential sweep between 2, 4, 8, 16, 32)
#     - Data generation (Zipf's law)
#     - Depth (2 4 8 16) 
#     - Width ()
#     - Choice of optimization = (Adam, SGD later)
#     - Batch size + learn rate decay
#     - Sentence length = (N + memory)



HYPERPARAMETERS = OrderedDict({
    'algorithm': [
        'lstm',
        'cnn',
        'transformer'
    ],
    'depth': [2, 4, 8, 16], # TODO DETERMINE
    'width': [16, 32,128, 256, 512, 1024],
    'sequence_dependence': [2,4,8,16],
    'vocab': [5000],
    'hmm_hidden': [100],
    'batch_size': [32],
    'bttp_len': [32],
})


DEFAULT_VALUES_SPEC = {
    'device': 'cuda',
    'lr': 0.0002,
    'max_step': 100000,
    'eta_min': 0.0,
    'embedding_dim': 256,
    'momentum': 0.0,
    'scheduler_type': 'constant',
    'optimizer_type': 'adam',
    'warmup_step': 0,
    'bptt_len': 28
}

REQUIRED_ATTRIBUTES = set(
    list(DEFAULT_VALUES_SPEC.keys()) +
    list(HYPERPARAMETERS.keys())
)
