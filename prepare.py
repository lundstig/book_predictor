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
    return torch.cat([torch.tensor(model.get_word_vector(w)) for w in fasttext.tokenize(desc)])


def build_embeddings_for_books(books, count):
    model = get_embeddings_model()
    X = []
    for book in tqdm(books[:count]):
        X.append(description_to_tensor(model, book.description))
    Y = [torch.tensor([[book.avg_rating]]) for book in books[:count]]
    torch.save(X, "data/X.bin")
    torch.save(Y, "data/Y.bin")


books = data.load_valid_books()
build_embeddings_for_books(books, 30000)
