import torch

print("Loading data...")
X = torch.load("data/X.bin")
Y = torch.load("data/Y.bin")

fraction = 0.1
total = int(len(Y) * fraction)
X_small = X[:total]
Y_small = Y[:total]

torch.save(X_small, "data/X_small.bin")
torch.save(Y_small, "data/Y_small.bin")