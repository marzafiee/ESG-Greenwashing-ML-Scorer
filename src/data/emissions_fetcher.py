import pandas as pd
import os

# loading Our World In Data CO2 Emissions CSV file
df = pd.read_csv("data/raw/owid-co2-data.csv")

# use this path if running from this directory: ../../data/raw/owid-co2-data.csv

# inspecting what we have
print(df.head(10))
print("-" * 20)
print(df.info())  # checks for data types and if there are any missing values

"""
After examining the dataset, we see that it is country-level, not company-level so we won't filter by company name. Instead we'll filter by country and year, then use it as a benchmark to cross-reference against what a company claims about emissions in that country/sector.

For our ESG scorer, these are the columns that matter to us:
Core identifiers: country, iso_code, year

And the most useful emissions columns would be:
co2 — total CO2 in million tonnes
co2_growth_prct — year on year % change (great for fact-checking "we reduced emissions by X%")
co2_per_capita
methane
nitrous_oxide
total_ghg — all greenhouse gases combined
"""


def fetch_emissions_by_country(country_name, start_year):
    """
    Returns a clean dataframe of emissions history given a country name and a year range.
    """
    # filtering for recent years at this stage
    recent_year = df[df["year"] >= start_year]

    # then for those filtered years, we are choosing to display the selct columns of interest
    filtered_columns = recent_year.loc[
        recent_year["country"] == country_name,
        [
            "country",
            "iso_code",
            "year",
            "co2",
            "co2_growth_prct",
            "co2_per_capita",
            "methane",
            "nitrous_oxide",
            "total_ghg",
        ],
    ]

    # saving to csv in data/raw/
    # making sure it exists
    os.makedirs("data/raw", exist_ok=True)
    safe_name = country_name.lower().replace(" ", "_")

    # save to csv file
    outputPath = f"data/raw/emissions_{safe_name}_from_{start_year}.csv"
    filtered_columns.to_csv(outputPath, index=False)
    print(f"Saved emissions data for {country_name} to {outputPath}")

    # returning the dataframe so other modules can use it
    return filtered_columns


# test
fetch_emissions_by_country("United States", 2005)
