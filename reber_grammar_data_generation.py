import pandas as pd
import random

# generating embedded Reber Grammar sequence
# detailed on page 1751 of the following paper
# https://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735
def gen_reber_seq():
    df0 = pd.DataFrame({"emission":["T", "P"], "next_state":[1, 2]})
    df1 = pd.DataFrame({"emission":["S", "X"], "next_state":[1, 3]})
    df2 = pd.DataFrame({"emission":["T", "V"], "next_state":[2, 4]})
    df3 = pd.DataFrame({"emission":["X", "S"], "next_state":[2, 5]})
    df4 = pd.DataFrame({"emission":["P", "V"], "next_state":[3, 5]})

    dfs = [df0, df1, df2, df3, df4]

    head_tail = random.sample(["P", "T"], 1)[0]
    reber_seq = [head_tail]
    states = []
    current_state = 0
    while(current_state != 5):
        states.append(current_state)
        row = dfs[current_state].sample(n=1, replace=False)
        observed = row.iloc[0]["emission"]
        current_state = row.iloc[0]["next_state"]
        reber_seq.append(observed)
    reber_seq.append(head_tail)
    return(reber_seq)
