from markov_data_generator import *

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# a toy example
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
seed = 0
np.random.seed(seed)

dep = 2
obs = 5
lag_size = 10

# initialize the length of the sequence
T = np.random.randint(lag_size + 1, 10 * lag_size)

# HMM specific
hid = 3
# initialize the probability tables
start_prob = initialize_start_prob(num_states=hid)
prev_hidden_state_seq = np.array([get_state(start_prob) for _ in range(lag_size)], dtype=int)
trans = init_trans_matrix(dep, hid, hid)
emission = rand_init_prob_matrix(hid, obs)

hmm_seq_data = hmm_generator_short_long(lag_size=lag_size, output_length=T,
                                    prev_hidden_state_seq=prev_hidden_state_seq, transition_matrix=trans,
                                    emission_matrix=emission)
print(hmm_seq_data)


# MC specific
# In markov chain the transition goes directly from obs to obs.
start_prob = initialize_start_prob(num_states=obs)
prev_state_seq = np.array([get_state(start_prob) for _ in range(lag_size)], dtype=int)
trans = init_trans_matrix(dep, obs, obs)

mc_seq_data = mc_generator_short_long(lag_size=lag_size, output_length=T,
                                    prev_state_seq=prev_state_seq, transition_matrix=trans)
print(mc_seq_data)
