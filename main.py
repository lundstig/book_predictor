import learning
import plotting
import torch
import time

TRAINING_PROPORTION = 0.1
VALIDATION_PROPORTION = 0.1
assert TRAINING_PROPORTION + VALIDATION_PROPORTION < 1

print("Loading data...")
X = torch.load("data/X.bin")
Y = torch.load("data/Y.bin")

print(len(X))
print(len(Y))
time.sleep(5)
