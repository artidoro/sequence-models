import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import functools
import time
import sys

# sys.path.append(os.path.dirname(os.getcwd()) + '/utils')

# import data_utils

# haven't researched whether there's a good structure for sequence data cnn
# don't know whether to add pooling layer or not and what is a good design
# haven't tested with the optimizer settings yet

'''
# ---------------------------------------------------------------------------- #
# helper functions
# ---------------------------------------------------------------------------- #

# one hot encoding of the input sequence
def one_hot(nclass, seq):
    identity = np.eye(nclass)
    nested = [identity[seq[i]] for i in range(len(seq))]
    result = np.array(nested).reshape(nclass*len(seq),)
    return(result)

# compute output dimension of convolution and pooling
def get_output_dim(input_len, k, p, s):
    output = (input_len + p - k) / s + 1
    # print(output)
    # assert (round(output, 0) == output)
    return int(output)

# Loads training/testing data from some folder
# folder: folder containing data files to load
# prefix: file prefix to identify files to load
# the number of input layer neurons
def load_data(folder, fileprefix, input_len):
    files = glob.glob("{}/{}/{}".format(os.getcwd(), folder, fileprefix))
    acc_input = []
    acc_target = []
    for j,f in enumerate(files):
        print(j)
        # load one data file
        one_seq = np.loadtxt(f, dtype=int).reshape(-1,)

        reformed_input = np.empty((len(one_seq) - input_len, 1, input_len), dtype=int)
        reformed_target = np.empty((len(one_seq) - input_len,), dtype=int)

        # slide a window to use the [t-input_len:t-1] sequence to predict the t-th
        for i in range(reformed_input.shape[0]):
            reformed_input[i, :, :] = one_seq[i: i + input_len]
            reformed_target[i] = one_seq[i + input_len]
        acc_input.append(reformed_input)
        acc_target.append(reformed_target)
    return (acc_input, acc_target)
'''

# ---------------------------------------------------------------------------- #
# Sequence model
# ---------------------------------------------------------------------------- #
class toyCNN(SequenceModel):

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

    def init_model(self, input_len=100, vocab, embedding_dim, n_layers, conv_kernel_width=5, num_cov_filter=6,
                   n_layers):
        model = Net(vocab_size=len(vocab), input_len=100, conv_kernel_width=5,
                num_conv_filter=6, pool_kernel_width=2, embedding_dim=embedding_dim, stride=1,
                padding=0)
        print(model)
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        return model

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def predict(self, inputs, padding=True):

        self.model.eval()

        # Evaluation
        with torch.no_grad():
            X = self.to_var(inputs)
            log_probs = self.model(X)
            probs = torch.exp(log_probs)

        # Switch back to the training mode
        self.model.train()

        return probs

    def train_step(self, inputs, targets, train_step=0):

        X = self.to_var(inputs)
        Y = self.to_var(targets[:, -1])

        pred = self.model(X)
        loss = nn.NLLLoss()(pred, Y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update scheduler
        self.update_scheduler(train_step)

        return loss

# ---------------------------------------------------------------------------- #
# Network structure
# ---------------------------------------------------------------------------- #

class Net(nn.Module):

    def __init__(self, vocab_size, input_len, conv_kernel_width, num_conv_filter,
                 pool_kernel_width, embedding_dim, stride, padding):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(1, num_conv_filter, conv_kernel_width * embedding_dim,
                               stride=stride, padding=padding)
        self.pool = torch.nn.MaxPool1d(kernel_size=pool_kernel_width,
                                       stride=pool_kernel_width, padding=0)
        out_conv_size = get_output_dim(input_len * embedding_dim,
                                       conv_kernel_width * embedding_dim,
                                       padding, stride)
        out_pool_size = get_output_dim(out_conv_size,
                                       pool_kernel_width,
                                       0, pool_kernel_width)
        self.fc1 = nn.Linear(num_conv_filter * out_pool_size, vocab_size)

    def forward(self, x):
        # print("initial shape")
        # print(x.shape)
        embeds = self.embeddings(x).view(x.shape[0], x.shape[1], -1)
        # print(embeds.shape)
        x = self.conv1(embeds)
        # print("shape after conv1")
        # print(x.shape)
        x = self.pool(x)
        # print("shape after pool")
        # print(x.shape)
        x = F.relu(x)
        # print("shape after relu")
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        # print("shape after fc1")
        # print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

'''
# load training data files into a matrix of num_training_instance x 1 x input_len
# num_training_instance = num_sequences x (len(one_seq) - input_len)
acc_input, acc_target = load_data(folder = "training_seq",
                                  # fileprefix = "hmm_3_hid_5_obs_100_lag_500_len_*",
                                  fileprefix = "V1hmm_hidden_5_lag_2_vocab_10_seqlen_50000_wordsline_64*",
                                  input_len=100)

data_input = functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), acc_input)
print("finish")
data_target = functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), acc_target)
print("finish")

train_size = int(0.5 * data_input.shape[0])
train_idx = np.random.choice(range(data_input.shape[0]), (train_size,), replace=False)
test_idx = np.setdiff1d(np.array(range(data_input.shape[0])), train_idx)

training_input = data_input[train_idx]
training_target = data_target[train_idx]
testing_input = data_input[test_idx]
testing_target = data_target[test_idx]

# compute embedding of the training input
# vocab = np.unique(training_input)
# assume we know the vocabulary size
vocab = range(10)
word_to_ix = {word: i for i, word in enumerate(vocab)}
embedding_dim = 5

# vocab = range(5000)
# word_to_ix = {word: i for i, word in enumerate(vocab)}
# embedding_dim = 300
# initialize network
net = Net(vocab_size=len(vocab), input_len=100, conv_kernel_width=5,
          num_conv_filter=6, pool_kernel_width=2, embedding_dim=embedding_dim, stride=1,
          padding=0)
print(net)

start_time = time.time()
# training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
batch_size = 32
num_batches = -(-(len(training_target) // batch_size))
num_batch_to_print = 100
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    # for i in range (1):
    for i in range (num_batches):
        idx = np.random.randint(0, len(training_target), (batch_size,))
        inputs = torch.from_numpy(training_input[idx])
        labels = torch.from_numpy(training_target[idx])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # print statistics
        if (i % num_batch_to_print == 0 and i != 0):
            print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / num_batch_to_print))
            running_loss = 0.0
print("finish training in {}".format(time.time() - start_time))

outputs = net(torch.from_numpy(testing_input))
print("finish")
_, predicted = torch.max(outputs, 1)

print("prediction accuracy = {:.3f}".format(sum(predicted == torch.from_numpy(testing_target)).item() / len(predicted)))
'''
