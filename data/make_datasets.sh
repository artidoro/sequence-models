#!/bin/bash
python3 markov_data_generator.py --vocab_size 5000 --hidden_size 100 --sequence_len 10000
python3 markov_data_generator.py --vocab_size 10000 --hidden_size 100 --sequence_len 10000
