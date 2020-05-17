import torch
import torch.nn as nn

GENRES=7

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=GENRES, layers=4):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=0.5)
        self.lstm_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp):
        # LSTM expects 3d tensor
        inp_reshaped = inp.view(inp.shape[0], 1, -1)
        lstm_output, (h_n, c_n) = self.lstm(inp_reshaped)
        output = self.lstm_to_out(h_n[-1])[0]
        return output

class BatchedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=GENRES, layers=1, max_words=100):
        super(BatchedModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=0.5)
        self.lstm_to_out = nn.Linear(hidden_dim, output_dim)

        self.input_dim = input_dim
        self.max_words = max_words

    def forward(self, X):
        X_lengths = [len(x) for x in X]
        cutoff = min(self.max_words, max(X_lengths))
        X_lengths = [min(len(x), cutoff) for x in X]
        padded_X = torch.zeros(len(X), cutoff, X[0].shape[1])
        for i, x in enumerate(X):
            padded_X[i][:X_lengths[i]] += x[:X_lengths[i]]
        X = nn.utils.rnn.pack_padded_sequence(padded_X, X_lengths, batch_first=True, enforce_sorted=False)

        lstm_output, (h_n, c_n) = self.lstm(X)
        last_hidden = h_n[-1]
        output = self.lstm_to_out(last_hidden)

        return output
