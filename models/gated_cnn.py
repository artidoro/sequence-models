import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequence_model import SequenceModel
from torch.autograd import Variable



class GatedCNNModel(nn.Module):
    '''
        PyTorch implementation of "Language Modeling with Gated Convolutional Networks" (https://arxiv.org/abs/1612.08083)
        adapted from from https://github.com/linzehui/Gated-Convolutional-Networks

        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self, vocab_size, embed_dim, kernel_width, n_layers, out_chs, res_block_count, dropout=0.0):
        super().__init__()
        self.res_block_count = res_block_count
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.padding_left = nn.ConstantPad1d((kernel_width - 1, 0), 0)

        self.conv_0 = nn.Conv1d(in_channels=embed_dim, out_channels=out_chs,
                                kernel_size=kernel_width)
        self.b_0 = nn.Parameter(torch.zeros(out_chs, 1))  # same as paper
        self.conv_gate_0 = nn.Conv1d(in_channels=embed_dim, out_channels=out_chs,
                                     kernel_size=kernel_width)
        self.c_0 = nn.Parameter(torch.zeros(out_chs, 1))

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=out_chs, out_channels=out_chs,
                                              kernel_size=kernel_width)
                                    for _ in range(n_layers)])

        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(out_chs, 1))  # collections of b
                                    for _ in range(n_layers)])

        self.conv_gates = nn.ModuleList([nn.Conv1d(in_channels=out_chs, out_channels=out_chs,
                                                   kernel_size=kernel_width)
                                         for _ in range(n_layers)])

        self.cs = nn.ParameterList([nn.Parameter(torch.zeros(out_chs, 1))
                                    for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs, vocab_size)
        self.dropout = nn.Dropout(p=dropout)  # todo use dropout

    # conv1d Input: (N, Cin, Lin)
    # constantpad1d Input: (N,C,Win)  Output: (N,C,Wout)

    def forward(self, seq):
        # seq:(batch,seq_len)
        batch_size = seq.size(0)
        seq_len = seq.size(1)
        x = self.embedding(seq)  # x: (batch,seq_len,embed_dim)
        x.transpose_(1, 2)  # x:(batch,embed_dim,seq_len) , embed_dim equals to in_channel

        x = self.padding_left(x)  # x:(batch,embed_dim,seq_len+kernel-1)  #padding left with 0
        A = self.conv_0(x)  # A: (batch,out_chs,seq_len)   seq_len because of padding (kernel-1)
        A += self.b_0  # b_0 broadcast
        B = self.conv_gate_0(x)  # B: (batch,out_chs,seq_len)
        B += self.c_0

        h = A * torch.sigmoid(B)  # h: (batch,out_chs,seq_len)
        # todo: add resnet
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.convs, self.conv_gates)):
            h = self.padding_left(h)  # h: (batch,out_chs,seq_len+kernel-1)
            A = conv(h) + self.bs[i]
            B = conv_gate(h) + self.cs[i]
            h = A * torch.sigmoid(B)  # h: (batch,out_chs,seq_len+kernel-1)
            if i % self.res_block_count == 0:  # todo Is this correct?
                h += res_input
                res_input = h

        h.transpose_(1, 2)  # h:(batch,seq_len,out_chs)

        logic = self.fc(h)  # logic:(batch,seq_len,vocab_size)
        logic.transpose_(1,2)  # logic:(batch,vocab_size,seq_len) cross_entropy input:(N,C,d1,d2,..) C is num of class
        return logic



class GatedCNN(SequenceModel):

    # Default parameter values
    vocab_size = 2000
    seq_len = 60
    embedding_dim = 200
    n_layers = 10
    kernel_width = 5
    out_chs = 64
    res_block_count = 5


    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)


    def init_model(self, depth=10, width=500):
        self.n_layers = self.depth
        self.out_chs = self.width
        self.vocab_size = self.vocab

        model = GatedCNNModel(self.vocab_size, self.embedding_dim, self.n_layers, self.kernel_width, self.out_chs, self.res_block_count, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        return model


    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)


    def predict(self, inputs):
        """
        Gets predictions for the next token of a batch of sequences (as a distribution over vocab tokens).
        
        Arguments:
            inputs : a Tensor of shape (batch_size, input_seq_length)

        Returns:
            probs : a Tensor of shape (batch_size, vocab_size)
        """

        # Turn on evaluation mode
        self.model.eval()

        # Evaluation
        with torch.no_grad():
            X = self.to_var(inputs)
            batch_size = inputs.size(0)
            logits = self.model(X)
            logits = logits.reshape(batch_size, -1, self.vocab)
            #print(logits.shape)

        # Switch back to the training mode
        self.model.train()

        return logits

    
    def train_step(self, inputs, targets, mems=None):
        """
        Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """
        X = self.to_var(inputs)
        # Y = self.to_var(targets[:, -1]) # GatedCNN only expects a 1-D array of the next token in each sequence

        pred = self.model(X)
        
        loss = nn.CrossEntropyLoss(ignore_index=0)(pred.reshape(-1, self.vocab), targets.flatten())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

