"""
cross_referencer.py — For each ESG claim, find matching news/emissions evidence and classify it as SUPPORTS / CONTRADICTS / UNVERIFIED (with a confidence score).
This divergence signal is what the scoring model later turns into an ESG trust score.

Process:
    - For each claim from claim_extractor.py (data/processed/claims_*.csv), find the most relevant external evidence from news_fetcher.py and emissions_fetcher.py (data/raw/)
    - Then use a similarity/entailment model to judge whether that evidence supports or contradicts the claim.

This  turns raw claims + evidence into the divergence signals that feed the final greenwashing score.
"""

# importing model for semantic similarity
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

"""
importing model for NLI (Natural Language Inference) Verdicts
this model reads two pieces of text and decides the logical relationship between them.
- The Premise is the text we are treating as evidence (our external evidence: the news article)
- The Hypothesis is the statement we're testing (i.e. the company's claim)

NOTE:
We're asking "does this evidence entail the company's claim?"
"""
from transformers import pipeline

nli = pipeline("zero-shot-classification", model="cross-encoder/nli-MiniLM2-L6-H768")

# other imports
import csv
import os
import json
import numpy as np
import hashlib
import uuid

from datetime import date

# starter configurations
EMBED_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "roberta-large-mnli"
TOP_K = 10  # how many articles to check per claim
MIN_SIMILARITY = 0.3  # below this, an article isn't meaningful evidence for the claim
NLI_PARAGRAPHS = 8  # only feed the first N paragraphs of an article to the NLI model
OUTPUT_DIR = "data/processed/evidence"

# roberta-large-mnli returns these labels, we map them to our own verdict names as aforementioned
LABEL_TO_VERDICT = {
    "ENTAILMENT": "SUPPORTS",
    "CONTRADICTION": "CONTRADICTS",
    "NEUTRAL": "UNVERIFIED",
}


# helper function to handle deduplication / checkpointing, similar to the functionlity in claim_extractor.py
def hash_pair(claim_id, article_key):
    # hashing claim_id + article_key together so we can tell if THIS specific (claim, article) match has already been written to the evidence csv
    return hashlib.md5(f"{claim_id}::{article_key}".encode()).hexdigest()


# ~~~~~~~~ LOADING MODELS, CLAIMS AND ARTICLES ~~~~~~~
# function to load our models (only once!)
def load_models():
    """
    Load both models a single time and hand them back to the caller.
    """
    try:
        embedder = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        # most likely cause: no internet on first run, or sentence-transformers not installed
        raise RuntimeError(f"Could not load embedding model '{EMBED_MODEL}': {e}")

    try:
        # top_k=None => return scores for all three labels, not just the top one
        nli = pipeline("text-classification", model=NLI_MODEL, top_k=None)
    except Exception as e:
        raise RuntimeError(f"Could not load NLI model '{NLI_MODEL}': {e}")

    return embedder, nli


# function to load claims from our extracted claims .CSV file per company
def load_claims(claims_csv_path):
    """
    Read the claims CSV into a list of dicts, generating claim_id if missing.
    """
    if not os.path.exists(claims_csv_path):
        raise FileNotFoundError(f"Claims file not found: {claims_csv_path}")

    with open(claims_csv_path, newline="", encoding="utf-8") as f:
        claims = list(csv.DictReader(f))

    if not claims:
        print(f"Warning! Claims file '{claims_csv_path}' is empty.")
        return []

    # claim_text is the one column we truly can't function without
    if "claim_text" not in claims[0].keys():
        raise ValueError(
            f"Claims CSV is missing the 'claim_text' column. "
            f"Found columns: {list(claims[0].keys())}"
        )

    # claim_extractor.py doesn't output a claim_id column, so instead we will generate a stable one here from company + claim_text. stable = re-running on the same csv
    # gives the same ids, which keeps the checkpointing in cross_reference() working
    if "claim_id" not in claims[0].keys():
        print(
            "No claim_id column found so generating stable ids from company + claim_text."
        )
        for claim in claims:
            basis = f"{claim.get('company', '')}::{claim.get('claim_text', '')}"
            claim["claim_id"] = hashlib.md5(basis.encode()).hexdigest()[:8]

    return claims


def load_news(news_json_path):
    """
    Read the news JSON and keep only articles with usable full_text.
    """
    if not os.path.exists(news_json_path):
        raise FileNotFoundError(f"News file not found: {news_json_path}")

    with open(news_json_path, encoding="utf-8") as f:
        try:
            articles = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse '{news_json_path}' as JSON: {e}")

    # drop articles with missing/empty text so the encoder never sees None
    usable = [a for a in articles if a.get("full_text")]

    dropped = len(articles) - len(usable)
    if dropped:
        print(f"Skipped {dropped} article(s) with no full_text.")

    if not usable:
        print(f"Warning! No usable articles found in '{news_json_path}'.")

    return usable
