import torch
import torch.nn as nn
from rnn import RNN
from tqdm import tqdm
import multiprocessing


torch.set_num_threads(multiprocessing.cpu_count())


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


def rnn_train(X, Y, learning_rate, epochs):
    n_letters = 10  # prepare.n_letters()
    n_hidden = 128

    rnn = RNN(n_letters, n_hidden)
    hidden = torch.zeros(1, n_hidden)

    n = len(X)
    loss_history = []
    current_loss = 0
    count = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} ")
        for i in tqdm(range(n)):
            x = X[i]
            y = Y[i]
            output, loss = rnn_train_single(rnn, x, y, learning_rate)

            current_loss += loss
            count += 1
        loss_history.append(current_loss / n)
        print(loss_history)
        current_loss = 0

    return rnn, loss_history
