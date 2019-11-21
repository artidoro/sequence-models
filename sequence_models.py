import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        embedding = self.embed(x)
        lstm_out = self.lstm(embedding)
        return 