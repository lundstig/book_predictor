import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss_history(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.show()
