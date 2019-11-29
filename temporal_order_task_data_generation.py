import random

# return a pair of sequence and label of the temporal order task
# on page 1763 of the following paper
# https://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735
def gen_temporal_order_seq():
    seq_length = random.randint(100, 110)
    seq = [random.choice(["a", "b", "c", "d"]) for i in range(seq_length)]
    idx1 = random.randint(10, 20)
    idx2 = random.randint(50, 60)
    seq[idx1] = random.choice(["X", "Y"])
    seq[idx2] = random.choice(["X", "Y"])
    label = seq[idx1] + seq[idx2]
    return (seq, label)
