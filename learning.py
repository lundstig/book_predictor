import torch
import torch.nn as nn
import prepare
from rnn import RNN
from tqdm import tqdm


def rnn_train_single(rnn: RNN, x, y, learning_rate, criterion=nn.MSELoss()):
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    for i in range(x.size()[0]):
        output, hidden = rnn(x[i], hidden)

    loss = criterion(output, y)
    loss.backward()

    # Update paramteres based on gradient
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def rnn_train(X, Y, learning_rate, plot_every=100):
    n_letters = prepare.n_letters()
    n_hidden = 128

    rnn = RNN(n_letters, n_hidden, 1)
    hidden = torch.zeros(1, n_hidden)

    n = len(X)
    loss_history = []
    current_loss = 0
    for i in tqdm(range(n)):
        x = X[i]
        y = Y[i]
        output, loss = rnn_train_single(rnn, x, y, learning_rate)

        current_loss += loss
        if (i + 1) % plot_every == 0:
            loss_history.append(current_loss / plot_every)
            current_loss = 0

    return rnn, loss_history
