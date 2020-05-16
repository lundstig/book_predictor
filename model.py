import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=4):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=0.5)
        self.lstm_to_out = nn.Linear(hidden_dim, 1)

    def forward(self, inp):
        # LSTM expects 3d tensor
        inp_reshaped = inp.view(inp.shape[0], 1, -1)
        lstm_output, (h_n, c_n) = self.lstm(inp_reshaped)
        output = self.lstm_to_out(h_n[-1])
        return output

class BatchedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=4):
        super(BatchedModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=0.5)
        self.lstm_to_out = nn.Linear(hidden_dim, 1)
        self.input_dim = input_dim

    def forward(self, X):
        X_lengths = [len(x) for x in X]
        longest = max(X_lengths)
        padded_X = torch.zeros(len(X), longest, X[0].shape[1])
        for i, x in enumerate(X):
            padded_X[i][:X_lengths[i]] += x
        X = nn.utils.rnn.pack_padded_sequence(padded_X, X_lengths, batch_first=True, enforce_sorted=False)

        lstm_output, (h_n, c_n) = self.lstm(X)
        last_hidden = h_n[-1]
        output = self.lstm_to_out(last_hidden)



        return torch.flatten(output)
