import requests
import json
import os

# to import, use: from sec_fetcher import fetch_10k_filings


def fetch_10k_filings(CIK_num):
    """
    Returns a list of a company's 10-K filings with dates and accession numbers when given their ticker or name.
    """

    # sec requires 10-digit 0-padded CIKs soo:
    CIK_num = CIK_num.zfill(
        10
    )  # so that if a user enters say 320193, it becomes 0000320193

    # for our url
    url = f"https://data.sec.gov/submissions/CIK{CIK_num}.json"

    # sec also recommends using a User-Agent header
    headers = {"User-Agent": "inezannemarie@gmail.com"}

    # now we can get all the information needed from this r  Response object (below). this line sends a request
    try:
        # make the request
        r = requests.get(url, headers=headers)
        # check the status with r.raise_for_status()
        r.raise_for_status()
        # status_code = r.status_code

        # print final url
        print(r.url)

        data = r.json()  # gives a python dict of our response and stores it

        # exploring returned data: (to inspect structure)
        # print(data.keys())

        # going into the dictionary to extract the filings values
        # print(data["filings"].keys())

        # after examining the data, now we want to see the filings under the 10-K column for this company
        company_name = data["name"]
        recent = data["filings"]["recent"]

        forms = recent["form"]
        dates = recent["filingDate"]
        accessions = recent["accessionNumber"]

        filings_10k = []

        print(f"For: {company_name}, dates and accessions for 10-K are: ")
        # filtering for 10-K filings from this data
        for i in range(len(forms)):
            # filtering by each row in the 10-K column to get their forms, dates and accessions
            if forms[i] == "10-K":
                filings_10k.append(
                    {"forms": forms[i], "date": dates[i], "accession": accessions[i]}
                )

        # why? because all the lists are aligned by index so index of i will then refer to the same filing across board or across all lists

        # now we need to save the data to the raw folder in the data directory
        # making sure it exists
        os.makedirs("data/raw", exist_ok=True)

        # save to json file
        outputPath = f"data/raw/{company_name}({CIK_num})_10k_filings.json"

        with open(outputPath, "w") as f:
            json.dump(filings_10k, f, indent=4)

        print(f"Saves {len(filings_10k)} filings to {outputPath}")

        return filings_10k
    except requests.exceptions.HTTPError as e:
        # throws an exception or error message if the url doesn't exist for the inputted CIK number
        print(f"HTTP error occurred: {e}")

    except requests.exceptions.ConnectionError as e:
        print(f"Connection error (internet or URL issue): {e}")

    except requests.exceptions.Timeout as e:
        print(f"Request timed out: {e}")

    except requests.exceptions.RequestException as e:
        print(f"General request error: {e}")


# trials
print(fetch_10k_filings("1318606"))
