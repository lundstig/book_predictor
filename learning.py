import torch
import torch.nn as nn
import torch.optim as optim
from rnn import RNN
from model import Model, BatchedModel
from tqdm import tqdm
import multiprocessing


torch.set_num_threads(multiprocessing.cpu_count())

class Evaluator:
    def __init__(self, X, Y, loss_function=nn.MSELoss()):
        self.X = X
        self.Y = Y
        self.loss_function = loss_function

    def evaluate_model_single(self, model):
        model.eval()
        with torch.no_grad():
            ret = self.__evaluate_single(lambda x: model(x))
        model.train()
        return ret

    def evaluate_model_batched(self, model):
        model.eval()
        with torch.no_grad():
            ret = self.__evaluate_batched(lambda x: model(x))
        model.train()
        return ret

    def evaluate_constant(self, k):
        with torch.no_grad():
            return self.__evaluate_single(lambda _: torch.tensor([[k]]))

    def __evaluate_single(self, predict):
        total_loss = 0
        for x, y in zip(self.X, self.Y):
            total_loss += float(self.loss_function(predict(x), y))
        return total_loss/len(self.X)

    def __evaluate_batched(self, predict):
        return float(self.loss_function(predict(self.X), torch.tensor(self.Y)))/len(self.X)


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

def train_model(X, Y, hidden_dim, learning_rate, epochs, evaluator=None):
    n = len(X)
    input_dim = X[0].shape[1]

    loss_function = nn.MSELoss()
    model = Model(input_dim, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_history = []
    current_loss = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} ")
        for i in tqdm(range(n)):
            x = X[i]
            y = Y[i]
            model.zero_grad()

            prediction = model(x)
            loss = loss_function(prediction, y)
            current_loss += float(loss)

            loss.backward()
            optimizer.step()

        if not evaluator == None:
            print("Validation loss:", evaluator.evaluate_model_single(model))
        average_loss = current_loss / n
        loss_history.append(average_loss)
        print(f"Epoch {epoch} complete, current loss: {average_loss}")
        current_loss = 0

    return model, loss_history

def train_model_batched(X, Y, hidden_dim, learning_rate, epochs, batch_size=10, evaluator=None):
    n = len(X)
    input_dim = X[0].shape[1]

    batches = n//batch_size

    loss_function = nn.MSELoss()
    model = BatchedModel(input_dim, hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_history = []
    current_loss = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} ")
        for batch in tqdm(range(batches)):
            start = batch*batch_size
            end = start+batch_size
            x_batch = X[start:end]
            y_batch = torch.tensor(Y[start:end])
            model.zero_grad()

            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            current_loss += float(loss)

            loss.backward()
            optimizer.step()

        if not evaluator == None:
            print("Validation loss:", evaluator.evaluate_model_batched(model))
        average_loss = current_loss / n
        loss_history.append(average_loss)
        print(f"Epoch {epoch} complete, current loss: {average_loss}")
        current_loss = 0

    return model, loss_history