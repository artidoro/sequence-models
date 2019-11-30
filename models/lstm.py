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

    
    @abstractmethod
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
        # Todo: Finish LSTM predict @Arti
        raise NotImplementedError()

    
    @abstractmethod
    def train_step(self, batch):
        """Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """
        # TODO: Finish LSTM train_step @Arti
        # Make use of self.optimizer
        
        raise NotImplementedError()



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

    def forward(self, x, hidden):
        # TODO: Finish LSTM Forward function (@Arti)
        x = self.embedding(x)
        

