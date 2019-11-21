import random
from torchtext import data
from torchtext import datasets


def torchtext_batch_iterators(root_path, train_path, validation_path, test_path,
        batch_size=32, bptt_len=32, device=None, batch_first=True, repeat=False):
    """
    Returns iterators for training, test and validation data from a single data file.
    Assumes that the data is already numerical and separated by spaces.
    Each batch in the iterators returned has a `batch.text` and `batch.target` field for language modeling.

    The best way to understand more about torchtext is to look directly in the source code.

    Sample call: train, val, test = torchtext_batch_iterators('generated_data/',
        'train/V0hmm_hidden_100_lag_2_vocab_10000_seqlen_10000000_wordsline_64.txt',
        'validation/V0hmm_hidden_100_lag_2_vocab_10000_seqlen_100000_wordsline_64.txt',
        'test/V0hmm_hidden_100_lag_2_vocab_10000_seqlen_100000_wordsline_64.txt')

    Arguments:
        path {[type]} -- root path.

    Keyword Arguments:
        batch_size {int} -- (default: {32})
        bptt_len {int} -- The number of words in a line of a batch. (default: {32})
        device {[type]} -- What device to put the tensors on. (default: {None})
        batch_first {bool} -- Whether the first dimension of the batch should be the batch size (default: {True})
        repeat {bool} -- Whether to continue yielding after running over the entire dataset. (default: {False})
        seed {int} -- Random seed for the splitting of the data. (default: {0})
        test_split {float} -- The percentage of the data that should be used as a test set. (default: {0.01})
        val_split {float} -- The percentage of the data that should be used as a validation set. (default: {0.01})
    """

    def tokenize(text):
        return list(map(int, text.split(' ')))

    text_field = data.Field(sequential=True, batch_first=batch_first, tokenize=tokenize, use_vocab=False)

    train, validation, test  = datasets.LanguageModelingDataset.splits(path=root_path, train=train_path, validation=validation_path, test=test_path,
        text_field=text_field, newline_eos=False)

    train_iter, val_iter, test_iter = data.BPTTIterator.splits((train, validation, test), batch_size=batch_size, device=device, bptt_len=bptt_len, repeat=repeat)

    return train_iter, val_iter, test_iter
