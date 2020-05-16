from __future__ import annotations
import csv
from dataclasses import dataclass
import string
from typing import List


@dataclass
class Book:
    isbn: str
    title: str = None
    avg_rating: float = None
    description: str = None
    genres: str = None

    def merge_in(self, o: Book):
        if not self.title:
            self.title = o.title
        if not self.isbn:
            self.isbn = o.isbn
        if not self.avg_rating:
            self.avg_rating = o.avg_rating
        if not self.description:
            self.description = o.description
        if not self.genres:
            self.genres = o.genres


def load_csv(filename: str) -> List[Book]:
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            yield Book(
                title=row["title"],
                isbn=row["isbn13"],
                avg_rating=float(row["average_rating"]),
            )


def load_blurbs(filename) -> List[Book]:
    with open(filename) as f:
        while title := f.readline().strip():
            description = f.readline().strip()
            genres = f.readline().strip()
            # Sometimes there is no line for genres
            if genres[0] in string.digits:
                isbn = genres
                genres = None
            else:
                isbn = f.readline().strip()
            yield Book(title=title, description=description, genres=genres, isbn=isbn)


def load_ratings(filename) -> List[Book]:
    with open(filename) as f:
        while isbn := f.readline().strip():
            rating = float(f.readline().strip())
            yield Book(isbn=isbn, avg_rating=rating)


def merge_booklists(*booklists: List[List[Book]]) -> List[Book]:
    books = {}
    for booklist in booklists:
        for book in booklist:
            key = book.isbn
            if key in books:
                books[key].merge_in(book)
            else:
                books[key] = book
    return list(books.values())


def load_all_books() -> List[Book]:
    print("Loading books...")
    books1 = load_csv("data/goodreads1.csv")
    books2 = load_blurbs("data/blurbs.txt")
    books3 = load_ratings("data/ratings.txt")

    all_books = merge_booklists(books1, books2, books3)
    print(f"Have {len(all_books):,} books in total")
    return all_books


def load_valid_books() -> List[Book]:
    valid_description = lambda book: book.description
    valid_genres = lambda book: book.genres
    all_books = load_all_books()
    with_desc = len(list(filter(valid_description, all_books)))
    with_genres = len(list(filter(valid_genres, all_books)))
    books = list(
        filter(lambda book: valid_description(book) and valid_genres(book), all_books)
    )
    print(
        f"There are {with_desc:,} books with description, and {with_genres:,} books with genres"
    )
    print(f"This results in {len(books):,} valid books")
    return books
