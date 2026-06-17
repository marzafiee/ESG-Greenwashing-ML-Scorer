import requests
import json
import os
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# to import, use: from news_fetcher import fetch_news_articles

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# realistic browser header so news sites don't block our article fetches
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_article_text(url: str) -> str:
    """
    Fetches and extracts the main body text from a news article URL.
    Returns an empty string if the fetch or parse fails.
    """
    if not url:
        return ""

    try:
        r = requests.get(url, headers=_BROWSER_HEADERS, timeout=15)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # join all paragraph text — filters out short nav/caption snippets
        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        ]

        return "\n\n".join(paragraphs)

    except Exception:
        return ""


def fetch_news_articles(company_name):
    """
    Returns a list of a recent news articles about that company's ESG/sustainability practices when given their name.
    """
    url = "https://newsapi.org/v2/everything"
    headers = {"Authorization": f"Bearer {NEWSAPI_KEY}"}

    # multiple ESG-focused queries per company — broader coverage than a single "ESG" search
    queries = [
        f"{company_name} ESG",
        f"{company_name} emissions 2025",
        f"{company_name} renewable energy",
        f"{company_name} sustainability",
        f"{company_name} carbon footprint",
        f"{company_name} workplace safety",
        f"{company_name} environment",
    ]

    try:
        news_articles = []
        seen_urls = set()

        for query in queries:
            params = {
                "q": query,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 100,
            }

            # make the request
            r = requests.get(url, headers=headers, params=params)
            # check the status with r.raise_for_status()
            r.raise_for_status()

            # print final url
            print(r.url)

            data = r.json()  # gives a python dict of our response and stores it
            articles_data = data.get("articles", [])
            print(f'Query "{query}": {len(articles_data)} articles fetched')

            # after examining the data, now we want to see the articles relevant to this company
            for article in articles_data:
                article_url = article.get("url")
                if article_url and article_url in seen_urls:
                    continue
                if article_url:
                    seen_urls.add(article_url)

                news_articles.append(
                    {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article_url,
                        "publishedAt": article.get("publishedAt"),
                    }
                )
        print(
            f"For: {company_name}, fetched {len(news_articles)} unique news articles after deduplication"
        )

        # fetch full article text for each unique result — 1s delay between requests to avoid rate limits
        total = len(news_articles)
        for i, article in enumerate(news_articles):
            print(f"Fetching full text for article {i + 1}/{total}...")

            full_text = fetch_article_text(article.get("url"))
            if not full_text:
                print(
                    f"Warning: could not fetch full text for article {i + 1}, falling back to description"
                )
                full_text = article.get("description") or ""

            article["full_text"] = full_text

            if i < total - 1:
                time.sleep(1)

        # now we need to save the data to the raw folder in the data directory
        # making sure it exists
        os.makedirs("data/raw", exist_ok=True)

        safe_name = company_name.lower().replace(" ", "_")
        # save to json file
        outputPath = f"data/raw/news_{safe_name}.json"

        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(news_articles, f, indent=4)

        print(f"Saves {len(news_articles)} news articles to {outputPath}")

        return news_articles

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return []

    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return []


# trials
print(fetch_news_articles("Apple"))
