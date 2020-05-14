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
