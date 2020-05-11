import data
import learning
import plotting
import prepare

TRAINING_PROPORTION = 0.1
VALIDATION_PROPORTION = 0.1
assert TRAINING_PROPORTION + VALIDATION_PROPORTION < 1

books = data.load_valid_books()

training_count = int(len(books) * TRAINING_PROPORTION)
validation_count = int(len(books) * VALIDATION_PROPORTION)
test_count = len(books) - training_count - validation_count
training_books = books[0:training_count]
validation_books = books[training_count : training_count + validation_count]
test_books = books[training_count + validation_count :]

print(
    f"Have {training_count:,} books for training, {validation_count:,} for validation and {test_count:,} for testing"
)

print("Building training data...")
X, Y = prepare.data_from_books(training_books)
rnn, loss_history = learning.rnn_train(X, Y, 0.005)

plotting.plot_loss_history(loss_history)
