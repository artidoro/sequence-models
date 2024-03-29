import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import functools

input_len = 100
num_filter_1 = 6
num_filter_2 = 16
num_obs = 5
kernel_width = 5 * num_obs
stride = 1
padding = 0

# haven't researched whether there's a good structure for sequence data cnn
# don't know whether to add pooling layer or not and what is a good design
# haven't tested with the optimizer settings yet
# may need embedding for when num_obs (or volcabulary) size is large

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input sequence channel, num_filter_1 output channels
        self.conv1 = nn.Conv1d(1, num_filter_1, kernel_width,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(num_filter_1, num_filter_2, kernel_width,
                               stride=stride, padding=padding)
        # out_pool_1_size = int(((input_len + padding - kernel_width) / stride + 1)/2)
        # print("out_pool_1_size {}".format(out_pool_1_size))
        # out_pool_2_size = int(((out_pool_1_size + padding - kernel_width) / stride + 1)/2)
        # print("out_pool_2_size {}".format(out_pool_2_size))
        # self.fc1 = nn.Linear(num_filter_2 * out_pool_2_size, 120)
        # fixme: hard-coded
        out_pool_2_size = 452
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(num_filter_2 * out_pool_2_size, 120)
        self.fc2 = nn.Linear(120, 84)
        # 5 observed states in hmm
        self.fc3 = nn.Linear(84, num_obs)

    def forward(self, x):
        # Max pooling over a (1, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net().float()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

# ---------------------------------------------------------------------------- #
# helper functions
# ---------------------------------------------------------------------------- #

# one hot encoding of the input sequence
def one_hot(nclass, seq):
    identity = np.eye(nclass)
    nested = [identity[seq[i]] for i in range(len(seq))]
    result = np.array(nested).reshape(nclass*len(seq),)
    return(result)

# folder: folder containing data files to load
# prefix: file prefix to identify files to load
def load_data(folder, fileprefix):
    files = glob.glob("{}/{}/{}".format(os.getcwd(), folder, fileprefix))
    acc_input = []
    acc_target = []
    for j,f in enumerate(files):
        print(j)
        # load one data file
        one_seq = np.loadtxt(f, dtype=int)

        reformed_input = np.empty((len(one_seq) - input_len, 1, input_len), dtype=int)
        reformed_target = np.empty((len(one_seq) - input_len,), dtype=int)

        # slide a window to use the last input_len sequence to predict the next one
        for i in range(reformed_seq.shape[0]):
            reformed_input[i, :, :] = one_seq[i: i + input_len]
            reformed_target[i] = one_seq[i + input_len]

        one_hot_input = np.empty((len(one_seq) - input_len, 1, input_len * num_obs), dtype=np.double)
        # one_hot encoded input
        for i in range(reformed_seq.shape[0]):
            one_hot_input[i, :, :] = one_hot(num_obs, reformed_input[i, :, :].transpose()).transpose()

        acc_input.append(one_hot_input)
        acc_target.append(reformed_target)
    return (acc_input, acc_target)

# load all training data files
acc_input, acc_target = load_data(folder = "training_seq",
                                  fileprefix = "hmm_3_hid_5_obs_100_lag_500_len_*")

training_input = functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), acc_input)
print("finish")

training_target = functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), acc_target)
print("finish")

# training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
batch_size = 32
num_batches = -(-(len(training_target) // batch_size))
num_batch_to_print = 100
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range (num_batches):
        idx = np.random.randint(0, len(training_target), (batch_size,))
        inputs = torch.from_numpy(training_input[idx])
        labels = torch.from_numpy(training_target[idx])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if i % num_batch_to_print == 0:
            running_loss += loss.item()
            print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / num_batch_to_print))
            running_loss = 0.0

# load all testing data files
acc_input, acc_target = load_data(folder = "testing_seq",
                                  fileprefix = "hmm_3_hid_5_obs_100_lag_500_len_9*")

testing_input = functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), acc_input)
print("finish")
testing_target = functools.reduce(lambda x, y: np.concatenate((x, y), axis=0), acc_target)
print("finish")

outputs = net(torch.from_numpy(testing_input).float())
print("finish")
_, predicted = torch.max(outputs, 1)

print("prediction accuracy = {:.3f}".format(sum(predicted == torch.from_numpy(testing_target)).item() / len(predicted)))
