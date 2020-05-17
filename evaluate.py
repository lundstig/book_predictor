import torch
import model
from collections import defaultdict

print("Loading model...")
model = torch.load("data/model.bin")

print("Loading data...")
X = torch.load("data/X.bin")
Y = torch.load("data/Y.bin")

TRAINING_PROPORTION = 0.5
VALIDATION_PROPORTION = 0.1
assert TRAINING_PROPORTION + VALIDATION_PROPORTION <= 1

total_count = len(X)
training_count = int(total_count * TRAINING_PROPORTION)
validation_count = int(total_count * VALIDATION_PROPORTION)
test_count = total_count - training_count - validation_count

start = training_count + validation_count
end = start + test_count
testX = X[start:end]
testY = Y[start:end]

model.eval()

genres = ['romance', 'young adult', 'science fiction', 'fantasy', 'history', 'mystery', 'biography']
matrix = []
for g in genres:
    matrix.append(defaultdict(lambda: defaultdict(int)))

with torch.no_grad():
    predicted = model(testX)
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            guess = bool(torch.sigmoid(predicted[i][j]) >= 0.5)
            actual = bool(testY[i][j] > 0.5) # is either 0 or 1
            matrix[j][guess][actual] += 1

for i, m in enumerate(matrix):
    TP = m[True][True]
    FP = m[True][False]
    TN = m[False][False]
    FN = m[False][True]

    if TP + FP + TN + FN == 0:
        print(genres[i], "does not exist")
        continue

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = -1
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = -1

    F1 = precision * recall / (precision + recall)

    print(f"{genres[i]}: {F1=:.2f} {accuracy=:.2f} {precision=:.2f}, {recall=:.2f}")
