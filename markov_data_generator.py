import numpy as np
import argparse
import os
from tqdm import tqdm


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# helper functions
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Generate an array of length num_states, and sum to 1
def initialize_start_prob(num_states):
    v = np.random.uniform(size=num_states)
    normalized = v/np.sum(v)
    return normalized

# Sample categorical RV
# prob: an np.array holding the PMF of a categorical RV
def get_state(prob):
    return np.nonzero(np.random.multinomial(n=1, pvals=prob))[0][0]

# Initialize a probability matrix each row has probabilities sum to one
# row name is a "from" state, col name is a "to" state
# result[i,j] is the probability of transition from state i to j
def rand_init_prob_matrix(nrow, ncol):
    return np.array([initialize_start_prob(ncol) for r in range(nrow)])

# Initialize a transition matrix from `number_dep_states` which can take
# `number_states_from` values to a new state which can take `number_states_to`
# values.
# Assuming `number_dep_states = 2`, result[i,j,k] is the probability of
# transition from state i,j to k
# number_dep_states: number of states on which the next state depends on
# number_states_from: number of different values previous states can take
# number_states_to: number of different values next state can take
def init_trans_matrix(number_dep_states, number_states_from, number_states_to):
    matrix = rand_init_prob_matrix(number_states_from ** number_dep_states, number_states_to)
    return matrix.reshape(np.append(np.full(number_dep_states, number_states_from), number_states_to))

# Get the next hidden states from the previous state and a long-ago state
# prev_state: the hidden state at the (t-1) step
# long_ago_state: the hidden state at the (t-q) step
# be 100 for example).
# transition_matrix
def get_next_state(prev_state, long_ago_state, transition_matrix):
    assert prev_state is not None
    assert long_ago_state is None
    assert long_ago_state < transition_matrix.shape[0]
    assert prev_state < transition_matrix.shape[0]
    prob = transition_matrix[long_ago_state, prev_state]
    state = get_state(prob)
    return state

# Generate a sequence of length "output_length" using a markov chain
# with dependence of two previous states t-1 and t-`lag_size`.
# lag_size: the number of step ago for the long-ago state, should be more than 1.
# output_length: the length of the generated sequence
# prev_state_seq: the previous state sequence to generate the next words
# transition_matrix: the transition matrix
def mc_generator_short_long(lag_size, output_length,
                             prev_state_seq,
                             transition_matrix):
    assert lag_size > 1, 'lag size should be greater than 1.'
    assert lag_size <= prev_state_seq.size, 'prev_sequence too small for the lag chosen.'

    seq = np.hstack([prev_state_seq, np.empty(output_length, dtype = int)])

    for t in range(prev_state_seq.size, output_length + prev_state_seq.size):
        seq[t] = get_state(transition_matrix[seq[t-lag_size], seq[t-1]])

    # Output everything after the initial sequence.
    return seq[prev_state_seq.size:]

# Generate a sequence of length "output_length" using a hidden markov model
# with dependence of two previous states t-1 and t-`lag_size`.
# lag_size: the number of step ago for the long-ago state
# weight: for adjusting the strength of dependence on state t-1 vs t-q (q could
# output_length: the length of the generated sequence
# prev_hidden_state_seq: the previous hidden state sequence used to generate the following
# transition_matrix: the transition matrix size
# emission_matrix: the emission matrix
def hmm_generator_short_long(lag_size, output_length,
                             prev_hidden_state_seq,
                             transition_matrix, emission_matrix):

    assert lag_size <= prev_hidden_state_seq.size, 'prev_sequence size {} too small for the lag {} chosen.'.format(
        prev_hidden_state_seq.size, lag_size)
    assert lag_size > 1, 'lag size should be greater than 1.'

    seq_hidden = np.hstack([prev_hidden_state_seq, np.empty(output_length, dtype = int)])
    seq_obs = np.empty(output_length, dtype = int)

    # At each step generate the next hidden state
    # and then the next observed state based on the hidden state.
    for t in range(prev_hidden_state_seq.size, output_length + prev_hidden_state_seq.size):
        seq_hidden[t] = get_state(transition_matrix[seq_hidden[t-lag_size], seq_hidden[t-1]])

        seq_obs[t-prev_hidden_state_seq.size] = get_state(emission_matrix[seq_hidden[t]])
    return (seq_hidden[prev_hidden_state_seq.size:], seq_obs)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# Generate the training file.
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

###############
# Increase this as we change the generation protocol and push every time to github so that we can
# track the changes in how the data is generated.
GEN_VERSION = 0
###############

dep = 2
vocab_default = 10000
lag_min_default = 1
lag_max_default = 5
mc_default = False
hidden_size_default = 100
sequence_len_default = 10000000
words_per_line_default = 64
dest_folder_default = 'generated_data'

if __name__ == '__main__':
    description = ("Generate data with HMM or MC models with dependency on the previous step t-1 "
                   "and step t-lag. This code writes the data to a file with exponentially increasing lag size. "
                   "The output files have user specified number of words separated by spaces on each line. "
                   "Note that with MC models the parameter size increases exponentially with the vocab "
                   "size. With very large vocab size it is therefore better to use HMM generation which "
                   "increases exponentially with the hidden state size.")

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--lag_min', default=lag_min_default, type=int,
        help='the min lag size. log scale. (default {})'.format(lag_min_default))
    parser.add_argument('--lag_max', default=lag_max_default, type=int,
        help='the max lag size. log scale. (default {})'.format(lag_max_default))
    parser.add_argument('--vocab_size', default=vocab_default, type=int,
        help='the vocabulary size. (default {})'.format(vocab_default))
    parser.add_argument('--mc', default=mc_default, type=bool,
        help='whether to use mc generator. By default uses hmm.')
    parser.add_argument('--hidden_size', default=hidden_size_default, type=int,
        help='the size of the hidden state. (default {})'.format(hidden_size_default))
    parser.add_argument('--sequence_len', default=sequence_len_default, type=int,
        help='the number of words in each file. (default {})'.format(sequence_len_default))
    parser.add_argument('--words_line', default=words_per_line_default, type=int,
        help='the number of words per line. (default {})'.format(words_per_line_default))
    parser.add_argument('--dest_folder', default=dest_folder_default,
        help='the destination folder. (default {})'.format(dest_folder_default))
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)

    file_base = 'mc'
    if args.mc is False:
        file_base = 'hmm_hidden_{}'.format(args.hidden_size)

    # We need to generate a series of files that have increasing lag (between min and max).
    # Each should have length file len.
    for idx, lag in enumerate([2**exp for exp in range(args.lag_min, args.lag_max + 1)]):

        file_name = '{}/V{}{}_lag_{}_vocab_{}_seqlen_{}_wordsline_{}.txt'.format(dest_folder_default, GEN_VERSION,
            file_base, lag, args.vocab_size, args.sequence_len, args.words_line)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as out_file:

            # Code specific for Markov Chain generation.
            if args.mc:
                # Initialize the starting sequence uniformly at random.
                start_prob = initialize_start_prob(num_states=args.vocab_size)
                prev_state_seq = np.array([get_state(start_prob) for _ in range(lag)], dtype=int)
                # Initialize the transition matrix.
                transition_matrix = init_trans_matrix(dep, args.vocab_size, args.vocab_size)

                for i in tqdm(range(int(np.ceil(args.sequence_len/args.words_line)))):
                    next_sequence = mc_generator_short_long(lag, args.words_line,
                                        prev_state_seq, transition_matrix)
                    line = ' '.join(map(str, next_sequence)) + '\n'
                    out_file.write(line)
                    prev_state_seq = next_sequence

            # Code specific for HMM generation.
            else:
                # Initialize the starting hidden sequence uniformly at random.
                start_prob = initialize_start_prob(num_states=args.hidden_size)
                prev_hidden_state_seq = np.array([get_state(start_prob) for _ in range(lag)], dtype=int)

                # Initialize the transition matrix.
                transition_matrix = init_trans_matrix(dep, args.hidden_size, args.hidden_size)
                emission_matrix = rand_init_prob_matrix(args.hidden_size, args.vocab_size)

                for i in tqdm(range(int(np.ceil(args.sequence_len/args.words_line)))):
                    (next_hid_sequence, next_obs_sequence) = hmm_generator_short_long(lag, args.words_line,
                                        prev_hidden_state_seq, transition_matrix, emission_matrix)
                    line = ' '.join(map(str, next_obs_sequence)) + '\n'
                    out_file.write(line)
                    prev_hidden_state_seq = next_hid_sequence

        print('Done generating file {}/{}.'.format(idx + 1, args.lag_max + 1 - args.lag_min))
