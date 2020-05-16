import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss_history(training_loss, validation_loss):
    plt.figure()
    plt.plot(training_loss, 'b')
    plt.plot(validation_loss, 'r')
    plt.show()
