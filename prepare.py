import torch
import string

ALLOWED_CHARS = string.ascii_lowercase + string.digits + " ,.!?()/-"
c2i = {}
i2c = list()
for i, c in enumerate(ALLOWED_CHARS):
    c2i[c] = i
    i2c.append(c)


def n_letters():
    return len(ALLOWED_CHARS)


def filter_text(text: str):
    return [c for c in text.lower() if c in ALLOWED_CHARS]


def character_to_tensor(c):
    tensor = torch.zeros(1, len(ALLOWED_CHARS))
    tensor[0][character_to_index(c)] = 1
    return tensor


def line_to_tensor(line):
    line = filter_text(line)
    tensor = torch.zeros(len(line), 1, len(ALLOWED_CHARS))
    for li, letter in enumerate(line):
        tensor[li][0][c2i[letter]] = 1
    return tensor


def data_from_books(books):
    X = [line_to_tensor(book.description) for book in books]
    Y = [torch.tensor([[book.avg_rating]]) for book in books]
    return X, Y
