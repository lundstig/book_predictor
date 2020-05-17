import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss_history(datasets, tid):
    colors = ['b', 'r', 'g']
    plt.figure()
    legends = []
    for i, (label, training_loss, validation_loss) in enumerate(datasets):
      a, = plt.plot(training_loss, color=colors[i], label=f"{label} training")
      b, = plt.plot(validation_loss, color=colors[i], linestyle='--', label=f"{label} validation")
      legends += [a, b]
    plt.legend(handles=legends)
    plt.title(tid)
    plt.savefig("out/" + tid + ".pdf", transparent=True)
    plt.savefig("out/" + tid + ".png")
