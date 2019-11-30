import torch
import torch.nn as nn
import torch.nn.functional as F



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
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        # self.embd_size = embd_size

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

        self.fc = nn.Linear(out_chs*seq_len, ans_size)

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
        h = A * F.sigmoid(B)    # (bs, Cout, seq_len, 1)
        res_input = h # TODO this is h1 not h0

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * F.sigmoid(B) # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, -1) # (bs, Cout*seq_len)
        out = self.fc(h) # (bs, ans_size)
        out = F.log_softmax(out)

        return out


class GatedCNN(SequenceModel):

    # Default parameter values
    vocab_size = 2000
    seq_len = 21
    embd_size = 200
    n_layers = 10
    kernel = (5, embd_size)
    out_chs = 64
    res_block_count = 5
    batch_size = 64


    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)


    def init_model(self, depth=10, width=500):
        self.n_layers = self.depth
        self.out_chs = self.width
        self.kernel[1] = self.embd_size

        model = GatedCNN(self.seq_len, self.vocab_size, self.embd_size, self.n_layers, self.kernel, self.out_chs, self.res_block_count, self.vocab_size)
        self.model = model
        return model


    def predict(self, batch, padding=True):
        """
        Gets all one-step predictions for a batch of sentences.
        For each context window output the next word.
        seq[0:k] -> pred[k+1]
        and so we output seq_len - k predictions
        without padding
        and seq_len predictions
        with padding
        """
        raise NotImplementedError()

    
    def train_step(self, batch):
        """Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """
        raise NotImplementedError()






