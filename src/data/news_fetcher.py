import requests
import json
import os
from dotenv import load_dotenv

# to import, use: from news_fetcher import fetch_news_articles

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


def fetch_news_articles(company_name):
    """
    Returns a list of a recent news articles about that company's ESG/sustainability practices when given their name.
    """
    url = "https://newsapi.org/v2/everything"
    headers = {"Authorization": f"Bearer {NEWSAPI_KEY}"}
    params = {"q": f"{company_name} ESG", "language": "en", "sortBy": "relevancy"}

    try:
        # make the request
        r = requests.get(url, headers=headers, params=params)
        # check the status with r.raise_for_status()
        r.raise_for_status()
        # status_code = r.status_code

        # print final url
        print(r.url)

        data = r.json()  # gives a python dict of our response and stores it
        articles_data = data.get("articles", [])
        news_articles = []  # our result from the function

        # after examining the data, now we want to see the articles relevant to this company
        print(f"For: {company_name}, some fetched news articles are: ")

        for article in articles_data:
            news_articles.append(
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                }
            )

        # now we need to save the data to the raw folder in the data directory
        # making sure it exists
        os.makedirs("data/raw", exist_ok=True)

        safe_name = company_name.lower().replace(" ", "_")
        # save to json file
        outputPath = f"data/raw/{safe_name}_news_articles.json"

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
# print(fetch_news_articles("Tesla"))
