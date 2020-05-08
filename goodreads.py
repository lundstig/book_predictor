import requests
from typing import List
from time import time, sleep


def ensure_delay(seconds: int = 3):
    since_last = time() - ensure_delay.last_time
    if since_last < seconds:
        sleep(seconds - since_last)
    ensure_delay.last_time = time()


ensure_delay.last_time = 0


def query_ratings(isbns: List[str], api_key: str):
    # Not too many books at once, results in too long URL
    assert len(isbns) <= 100
    # Limit QPS to comply with goodreads API terms
    ensure_delay()

    parameters = {"key": api_key, "isbns": ",".join(isbns)}
    print(f"Getting ratings for {len(isbns)} books")
    r = requests.get(
        "https://www.goodreads.com/book/review_counts.json", params=parameters
    )
    books = r.json()["books"]
    return {book["isbn13"]: float(book["average_rating"]) for book in books}
