# ESG Finance ML Scorer

A machine learning project that scores companies on ESG credibility by comparing
what they claim in SEC filings against independent news sources and emissions databases.

## Project Status
In development.

## Data Sources
- SEC EDGAR (company filings)
- NewsAPI (independent news coverage)
- Our World in Data CO2 dataset (emissions ground truth)

## Setup
1. Clone the repo
2. Create a virtual environment and run `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run fetchers from `src/data/`

## Structure
    data/        raw and processed data files
    notebooks/   exploration and experiments
    src/         source modules
    reports/     generated company score reports
