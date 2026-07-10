# checking to see if fetch_article_text() actually pulls the entire article text from the url

# run this file with python tests/test_news_fetcher.py

import sys

sys.path.append("src/data")

from news_fetcher import fetch_article_text

test_urls = [
    "https://www.reuters.com/business/sustainable-business/",
    "https://www.theguardian.com/environment/climate-crisis/",
    "https://www.bbc.com/news/science-environment",
]

for url in test_urls:
    text = fetch_article_text(url)
    print(f"URL: {url}")
    print(f"Length: {len(text)}")
    print(f"Preview: {text[:200]}")
    print("---")
