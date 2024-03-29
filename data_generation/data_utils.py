import random
import numpy as np
from torchtext import data
from torchtext import datasets


def torchtext_batch_iterators_split(root_path, file_path, train_prefix='train', test_prefix='test',
        batch_size=32, bptt_len=32, device=None, batch_first=True, repeat=False, test_size=10000, words_line=64):
    """
    Returns iterators for training, test and validation data from a single data file.
    Assumes that the data is already numerical and separated by spaces.
    Each batch in the iterators returned has a `batch.text` and `batch.target` field for
    language modeling.

    The best way to understand more about torchtext is to look directly in the source code.

    Arguments:
        path {[type]} -- root path.

    Keyword Arguments:
        batch_size {int} -- (default: {32})
        bptt_len {int} -- The number of words in a line of a batch. (default: {32})
        device {[type]} -- What device to put the tensors on. (default: {None})
        batch_first {bool} -- Whether the first dimension of the batch should be the batch size (default: {True})
        repeat {bool} -- Whether to continue yielding after running over the entire dataset. (default: {False})int(np.ceil(args.sequence_len/args.words_line))
    """
    # Split the file into two files (train and validation). Size of the validation data is specified above.
    data_file_path = root_path + '/' + file_path
    train_file_path =  "train_"+ file_path
    test_file_path =  "test_" + file_path
    with open(data_file_path, 'r') as data_file, open(root_path + "/" +  train_file_path, 'w') as train_file, open(root_path + "/"+ test_file_path, 'w') as test_file:
        for idx, line in enumerate(data_file.readlines()):
            if idx < int(np.ceil(test_size/words_line)):
                test_file.write(line)
            else:
                train_file.write(line)

    def tokenize(text):
        x =  list(map(int, [x for x in text.split(' ') if x]))
        return x

    text_field = data.Field(sequential=True, batch_first=batch_first, tokenize=tokenize, use_vocab=False)

    train, test  = datasets.LanguageModelingDataset.splits(path=root_path, train=train_file_path,
        test=test_file_path, text_field=text_field, newline_eos=False)

    train_iter, test_iter = data.BPTTIterator.splits((train, test),
        batch_size=batch_size, device=device, bptt_len=bptt_len, repeat=repeat)



    return train_iter, test_iter


def torchtext_batch_iterators(root_path, train_path, test_path,
        batch_size=32, bptt_len=32, device=None, batch_first=True, repeat=False):

    def tokenize(text):
        return list(map(int, text.split(' ')))

    text_field = data.Field(sequential=True, batch_first=batch_first, tokenize=tokenize, use_vocab=False)

    train, test  = datasets.LanguageModelingDataset.splits(path=root_path, train=train_path,
        test=test_path, text_field=text_field, newline_eos=False)

    train_iter, test_iter = data.BPTTIterator.splits((train, test),
        batch_size=batch_size, device=device, bptt_len=bptt_len, repeat=repeat)

    return train_iter, test_iter