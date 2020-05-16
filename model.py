import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=2):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers)
        self.lstm_to_out = nn.Linear(hidden_dim, 1)

    def forward(self, inp):
        # LSTM expects 3d tensor
        inp_reshaped = inp.view(inp.shape[0], 1, -1)
        lstm_output, (h_n, c_n) = self.lstm(inp_reshaped)
        output = self.lstm_to_out(h_n[-1])
        return output

class BatchedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=2):
        super(BatchedModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers)
        self.lstm_to_out = nn.Linear(hidden_dim, 1)
        self.input_dim = input_dim

    def forward(self, X):
        #print(X)
        #print(X[0].shape)
        #print(X[1].shape)
        #print(X[2].shape)

        X_lengths = [len(x) for x in X]
        longest = max(X_lengths)
        padded_X = torch.zeros(len(X), longest, X[0].shape[1])
        for i, x in enumerate(X):
            padded_X[i][:X_lengths[i]] += x
        #print("x lens", X_lengths)
        #print("px", padded_X.shape)
        X = nn.utils.rnn.pack_padded_sequence(padded_X, X_lengths, batch_first=True, enforce_sorted=False)
        #print("x_packed", X)

        lstm_output, (h_n, c_n) = self.lstm(X)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        #print("lstm_output", lstm_output)
        #print("unpadded", output)
        #print("hidden", hidden)
        #print("h_n shape", h_n[-1].shape)
        #print("output_shape", output.shape)
        last_hidden = h_n[-1]
        output = self.lstm_to_out(last_hidden)



        return torch.flatten(output)
