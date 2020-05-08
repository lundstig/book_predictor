import data
import goodreads

books = data.load_all_books()

without_rating = list(filter(lambda book: not book.avg_rating, books))
print(f"{len(without_rating):,} books are missing rating")
if not without_rating:
    exit(0)

print("Will try to download missing ratings")
api_key = input("Goodreads API key? ")

batch_size = 100
with open('data/ratings.txt', "a") as f:
    for i in range(0, len(without_rating), batch_size):
        batch = without_rating[i:i+batch_size]
        isbns = [book.isbn for book in batch]
        ratings = goodreads.query_ratings(isbns, api_key)
        for isbn, rating in ratings.items():
            f.write(isbn + "\n")
            f.write(str(rating) + "\n")
        
    f.flush()
