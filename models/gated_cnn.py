import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequence_model import SequenceModel
from torch.autograd import Variable



class GatedCNNModel(nn.Module):
    '''
        PyTorch implementation of "Language Modeling with Gated Convolutional Networks" (https://arxiv.org/abs/1612.08083)
        adapted from from https://github.com/jojonki/Gated-Convolutional-Networks

        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''

    def __init__(self,
                 seq_len,
                 vocab_size,
                 embd_size,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 ans_size):
        super().__init__()
        self.res_block_count = res_block_count
        self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs * seq_len, ans_size)
        self.seq_len = seq_len
        self.out_chs = out_chs


    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        seq_len = x.size(1)
        x = self.embedding(x) # (bs, seq_len, embd_size)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        h = A * torch.sigmoid(B)    # (bs, Cout, seq_len, 1)
        res_input = h # TODO this is h1 not h0

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * torch.sigmoid(B) # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, self.out_chs*seq_len) # (bs, Cout*seq_len)
        out = self.fc(h) # (bs, ans_size)
        out = F.log_softmax(out, dim=-1)

        return out


class GatedCNN(SequenceModel):

    # Default parameter values
    vocab_size = 2000
    seq_len = 60
    embedding_dim = 200
    n_layers = 10
    kernel = (5, embedding_dim)
    out_chs = 64
    res_block_count = 5


    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)


    def init_model(self, depth=10, width=500):
        self.n_layers = self.depth
        self.out_chs = self.width
        self.vocab_size = self.vocab
        self.kernel = (self.kernel[0], self.embedding_dim)
        self.seq_len = self.bttp_len

        model = GatedCNNModel(self.seq_len, self.vocab_size, self.embedding_dim, self.n_layers, self.kernel, self.out_chs, self.res_block_count, self.vocab_size)
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
            log_probs = self.model(X)
            probs = torch.exp(log_probs)

        # Switch back to the training mode
        self.model.train()

        return probs

    
    def train_step(self, inputs, targets, train_step=0, mems=None):
        """
        Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """
        X = self.to_var(inputs)
        Y = self.to_var(targets[:, -1]) # GatedCNN only expects a 1-D array of the next token in each sequence

        pred = self.model(X)
        loss = nn.NLLLoss()(pred, Y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update scheduler
        self.update_scheduler(train_step)

        return loss

