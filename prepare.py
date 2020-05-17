import data
import fasttext
import string
import torch
from tqdm import tqdm


def data_from_books(books):
    X = [line_to_tensor(book.description) for book in books]
    Y = [torch.tensor([[book.avg_rating]]) for book in books]
    return X, Y


def get_embeddings_model():
    return fasttext.load_model("data/wiki-news-300d-1M-subword.bin")


def description_to_tensor(model, desc):
    return torch.stack(
        [torch.tensor(model.get_word_vector(w)) for w in fasttext.tokenize(desc)]
    )

def genres_to_tensor(s):
    s = s.lower()
    genres = ['romance', 'young adult', 'science fiction', 'fantasy', 'histor', 'myster', 'biograph']
    ret = torch.zeros(len(genres))
    for i, genre in enumerate(genres):
        if genre in s:
            ret[i] = 1
    return ret


def build_embeddings_for_books(books, count):
    model = get_embeddings_model()
    X = []
    Y = []
    for book in tqdm(books[:count]):
        X.append(description_to_tensor(model, book.description)[:100])
        Y.append(genres_to_tensor(book.genres))
    torch.save(X, "data/X.bin")
    torch.save(Y, "data/Y.bin")


books = data.load_valid_books()
build_embeddings_for_books(books, 30000)
