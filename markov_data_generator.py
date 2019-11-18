import numpy as np

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# helper functions
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Generate an array of length num_states, and sum to 1
def initialize_start_prob(num_states):
    v = np.random.uniform(size=num_states)
    normalized = v/sum(v)
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
# starting_prob: a length num_hidden_states array to sample the starting hidden state from
# transition_matrix: the transition matrix
def mc_generator_short_long(lag_size, output_length,
                             starting_prob,
                             transition_matrix):
    assert lag_size > 1

    print("mc generating a data sequence of length {:d}".format(output_length))
    seq = np.empty(output_length + lag_size, dtype = int)

    # Initialize the sequence by independent draws from the `starting_prob`.
    for i in range(lag_size):
        seq[i] = get_state(starting_prob)

    for t in range(lag_size, output_length + lag_size):
        seq[t] = get_state(transition_matrix[seq[t-lag_size], seq[t-1]])

    # Output everything after the initial sequence.
    return seq[lag_size:]

# Generate a sequence of length "output_length" using a hidden markov model
# with dependence of two previous states t-1 and t-`lag_size`.
# lag_size: the number of step ago for the long-ago state
# weight: for adjusting the strength of dependence on state t-1 vs t-q (q could
# output_length: the length of the generated sequence
# starting_prob: a length num_hidden_states array to sample the starting hidden state from
# transition_matrix: the transition matrix size
# emission_matrix: the emission matrix
def hmm_generator_short_long(lag_size, output_length,
                             starting_prob,
                             transition_matrix, emission_matrix):
    print("hmm generating a data sequence of length {:d}".format(output_length))
    seq_hidden = np.empty(output_length + lag_size, dtype = int)
    seq_obs = np.empty(output_length, dtype = int)

    # Initialize the sequence by independent draws from the `starting_prob`.
    for i in range(lag_size):
        seq_hidden[i] = get_state(starting_prob)

    # At each step generate the next hidden state
    # and then the next observed state based on the hidden state.
    for t in range(output_length):
        seq_hidden[t+lag_size] = get_state(transition_matrix[seq_hidden[t],
            seq_hidden[t+lag_size-1]])
        seq_obs[t] = get_state(emission_matrix[seq_hidden[t+lag_size]])
    return seq_obs


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# a toy example
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
seed = 0
np.random.seed(seed)

dep = 2
hid = 3
obs = 5

lag_size = 10

# initialize the length of the sequence
T = np.random.randint(lag_size + 1, 10 * lag_size)

# HMM specific
# initialize the probability tables
start_prob = initialize_start_prob(num_states=hid)
trans = init_trans_matrix(dep, hid, hid)
emission = rand_init_prob_matrix(hid, obs)

hmm_seq_data = hmm_generator_short_long(lag_size=lag_size, output_length=T,
                                    starting_prob=start_prob, transition_matrix=trans,
                                    emission_matrix=emission)
print(hmm_seq_data)


# MC specific
# In markov chain the transition goes directly from obs to obs.
start_prob = initialize_start_prob(num_states=obs)
trans = init_trans_matrix(dep, obs, obs)

mc_seq_data = mc_generator_short_long(lag_size=lag_size, output_length=T,
                                    starting_prob=start_prob, transition_matrix=trans)
print(mc_seq_data)