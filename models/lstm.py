import torch
import torch
import torch.nn as nn
from models.sequence_model import SequenceModel

class LSTMModel(SequenceModel):

    def init_model(self):
        return LSTMModel(
            self.vocab,
            self.embedding_dim, 
            self.width, 
            self.depth,
            self.output)

    



class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim):
        super(LSTMModel, self).__init__()

        # TODO: Finish LSTM architecture (@Arti)

        # Hidden dimensions
        self.vocab_size =vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # TODO: Finish LSTM Forward function (@Arti)
        x = self.embedding(x)

