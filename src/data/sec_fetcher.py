"""
Returns a list of a company's 10-K filings with dates and accession numbers when given their ticker or name.
"""

import requests

# logic flow
# input - a company's CIK number (can be found on the SEC website)
CIK_num = input("Input CIK number for the company of interest: ")

# sec requires 10-digit 0-padded CIKs soo:
CIK_num = CIK_num.zfill(
    10
)  # so that if a user enters say 320193, it becomes 0000320193

# for our url
url = f"https://data.sec.gov/submissions/CIK{CIK_num}.json"

# sec also recommends using a User-Agent header
headers = {"User-Agent": "inezannemarie@gmail.com"}

# now we can get all the information needed from this r  Response object (below). this line sends a request
r = requests.get(url, headers=headers)

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

print(f"For: {company_name}, dates and accessions for 10-K are: ")
# filtering for 10-K filings from this data
for i in range(len(forms)):
    # filtering by each row in the 10-K column to get their forms, dates and accessions
    if forms[i] == "10-K":
        print(forms[i], dates[i], accessions[i])

# why? because all the lists are aligned by index so index of i will then refer to the same filing across board or across all lists
