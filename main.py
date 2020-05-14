import learning
import plotting
import torch

print("Loading data...")
X = torch.load("data/X_small.bin")
Y = torch.load("data/Y_small.bin")

TRAINING_PROPORTION = 0.2
VALIDATION_PROPORTION = 0.1
assert TRAINING_PROPORTION + VALIDATION_PROPORTION < 1

total_count = len(X)
training_count = int(total_count * TRAINING_PROPORTION)
validation_count = int(total_count * VALIDATION_PROPORTION)
test_count = total_count - training_count - validation_count

print(f"{training_count:,} training samples")
trainingX = X[:training_count]
trainingY = Y[:training_count]

validationX = X[training_count:][:validation_count]
validationY = Y[training_count:][:validation_count]
evaluator = learning.Evaluator(validationX, validationY)

def mean(Ys):
  return sum(Ys) / len(Ys)

def getMeanLoss(Ys):
    meanY = mean(Ys)
    loss = 0
    for y in Ys:
        loss += (y - meanY) ** 2
    return round(float(loss / len(Ys)), 4)

print("Loss from guessing mean: ", getMeanLoss(trainingY))
print("Evaluator loss on guessing mean:", evaluator.evaluate_constant(mean(trainingY)))
model, loss_history = learning.train_model(trainingX, trainingY, 100, 0.01, 3, evaluator)
print(loss_history)
plotting.plot_loss_history(loss_history)
