# fetches the actual document from the filings fetched from sec_fetcher.py
# the JSON files generated in particular

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
import os

# to import, use: from filing_fetacher import filing_fetcher

load_dotenv()  # loading the header value from .env


def filing_fetcher(sec_JSON_filename, accession_num):
    """
    Returns the actual 10-K document given the SEC JSON filename and accession number
    """

    headers = {"User-Agent": os.getenv("SEC_USER_AGENT")}
    # print("USER AGENT:", headers["User-Agent"])

    # extract CIK from filename - it's in parantheses
    # using the .find() string slicing approach in py
    text = sec_JSON_filename
    start = text.find("(") + 1
    end = text.find(")")
    CIK_num = text[start:end].zfill(10)
    # print(CIK_num)  # for testing purposes

    # get company name from SEC API
    meta_url = f"https://data.sec.gov/submissions/CIK{CIK_num}.json"
    meta = requests.get(meta_url, headers=headers).json()
    company_name = meta["name"]

    print(f"Fetching filings for: {company_name}")

    # remove dashes from accession number
    # eg: 0001628280-26-003952
    accession_no_dashes = re.sub(r"\D", "", accession_num)
    # print(accession_no_dashes)  # for testing purposes

    # building our URL to get the associated index page
    html_page_url = f"https://www.sec.gov/Archives/edgar/data/{CIK_num}/{accession_no_dashes}/{accession_num}-index.htm"

    # fetch the page first
    r = requests.get(html_page_url, headers=headers)

    # parse it with BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")
    # print(soup.prettify())

    # find all table rows
    rows = soup.find_all("tr")

    # find the row where Type == "10-K" (Seq 1)
    for row in rows:
        columns = row.find_all("td")

        # skip rows that don't have enough columns
        if len(columns) < 4:
            continue

        if columns[0].text.strip() == "1" and columns[3].text.strip() == "10-K":
            # extract the document filename (e.g. tsla-20251231.htm)
            link_tag = columns[2].find("a")
            href = link_tag[
                "href"
            ]  # eg output: /ix?doc=/Archives/edgar/data/.../tsla-20251231.htm

            # strip the XBRL viewer prefix if present
            if "?doc=" in href:
                href = href.split("?doc=")[
                    1
                ]  # keeping only /Archives/edgar/data/.../tsla-20251231.htm

            # construct full document URL and fetch it
            BASE_URL = "https://www.sec.gov/"
            full_url = BASE_URL + href

            print(full_url)

            # actually fetching the 10-K document
            doc_response = requests.get(full_url, headers=headers)

            # save raw HTML to data/raw/
            # making sure it exists
            os.makedirs("data/raw", exist_ok=True)

            # save to HTML file
            outputPath = (
                f"data/raw/filing_{company_name}_{CIK_num}_{accession_no_dashes}.html"
            )

            with open(outputPath, "w", encoding="utf-8") as f:
                f.write(doc_response.text)

            print(f"Saved filing HTML for {company_name} to {outputPath}")

            # return the 10-K document
            return doc_response.text


# testing
# filing_fetcher("sec_Tesla, Inc.(0001318605)_10k_filings.json", "0001628280-26-003952")
