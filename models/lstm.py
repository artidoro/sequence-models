import torch
import torch
import torch.nn as nn
from models.sequence_model import SequenceModel

import config as c

class LSTMModel(SequenceModel):
    # https://github.com/ceshine/examples/blob/master/word_language_model/main.py

    def init_model(self):
        return RNNModel(
            'LSTM',
            self.vocab,
            self.embedding_dim,
            self.hidden_dim,
            self.depth)

    def predict(self, inputs, padding=True):
        """
        Gets all one-step predictions for a batch of sentences.
        For each context window output the next word.
        seq[0:k] -> pred[k+1]
        and so we output seq_len - k predictions
        without padding
        and seq_len predictions
        with padding
        """
        batch_size = inputs.shape[0]
        self.model.eval()
        hidden = self.model.init_hidden(batch_size)
        output, hidden = self.model(inputs, hidden)

        return output
        
        
    def train_step(self, inputs, targets, train_step=0):

        """Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """
        # TODO: Refactor this to the current setting.
        # TODO: Make use of self.optimizer
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        self.model.train()
        total_loss = 0
        lossfn = nn.CrossEntropyLoss()

        hidden = self.model.init_hidden(batch_size)
        # for batch, i in enumerate(range(0, seq_len, self.bptt_len)):
        #     bp_seq_len = min(self.bptt_len, seq_len - i)
        #     inp = inputs[:, i:i+bp_seq_len]
        #     tar = targets[:, i:i+bp_seq_len]
        #     # Starting each batch, we detach the hidden state from how it was previously produced.
        #     # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = self.repackage_hidden(hidden)
        self.model.zero_grad()
        output, hidden = self.model(inputs, hidden)
        loss = lossfn(output.view(-1, self.vocab), targets.flatten())
        loss.backward()
        self.optimizer.step()

        # # TODO: UPDATE THIS TO RESPECT THE OPTIMIZERS STUFF>
        # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # for p in self.model.parameters(): # TODO: What is this for?
        #     p.data.add_(-self.lr, p.grad.data)

        total_loss += loss.item()

        # Update scheduler
        self.update_scheduler(train_step)

        return total_loss


    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)



class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
