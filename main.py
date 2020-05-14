import learning
import plotting
import torch

print("Loading data...")
X = torch.load("data/X.bin")
Y = torch.load("data/Y.bin")

TRAINING_PROPORTION = 0.01
VALIDATION_PROPORTION = 0.1
assert TRAINING_PROPORTION + VALIDATION_PROPORTION < 1

total_count = len(X)
training_count = int(total_count * TRAINING_PROPORTION)
validation_count = int(total_count * VALIDATION_PROPORTION)
test_count = total_count - training_count - validation_count

print(f"{training_count:,} training samples")
trainingX = X[:training_count]
trainingY = Y[:training_count]


model, loss_history = learning.train_model(trainingX, trainingY, 100, 0.01, 3)
print(loss_history)
plotting.plot_loss_history(loss_history)
