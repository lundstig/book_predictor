import learning
import plotting
import torch
import model

print("Loading data...")
X = torch.load("data/X.bin")
Y = torch.load("data/Y.bin")

desc_limit = 100
print(X[0].shape)
for i, x in enumerate(X):
  X[i] = x[:desc_limit]

zero_count = 0
for y in Y:
  if y.sum().item() == 0:
    zero_count += 1
print("Zero:", zero_count, "of", len(Y))

Y = torch.stack(Y)

TRAINING_PROPORTION = 0.5
VALIDATION_PROPORTION = 0.1
EPOCHS = 6
assert TRAINING_PROPORTION + VALIDATION_PROPORTION <= 1

total_count = len(X)
training_count = int(total_count * TRAINING_PROPORTION)
validation_count = int(total_count * VALIDATION_PROPORTION)
test_count = total_count - training_count - validation_count

print(f"{training_count:,} training samples")
trainingX = X[:training_count]
trainingY = Y[:training_count]

validationX = X[training_count:][:validation_count]
validationY = Y[training_count:][:validation_count]
with torch.no_grad():
    evaluator = learning.Evaluator(validationX, validationY)

def get_average(Y):
  s = torch.tensor([0.5]*model.GENRES)
  for y in Y:
    s += y
  return s/len(Y)

print("Average:", get_average(trainingY))
print("Loss guessing average:", evaluator.evaluate_constant(get_average(trainingY)))

model, training_loss_batched, validation_loss_batched = \
    learning.train_model_batched(trainingX, trainingY, 64, 0.01, EPOCHS, evaluator=evaluator)
print(training_loss_batched)
print(validation_loss_batched)

# model, training_loss, validation_loss = learning.train_model(trainingX, trainingY, 64, 0.01, EPOCHS, evaluator)
# print(training_loss)
# print(validation_loss)

torch.save(model, "data/model.bin")
plotting.plot_loss_history([
  ('batched', training_loss_batched, validation_loss_batched),
  # ('single', training_loss, validation_loss),
  ])
