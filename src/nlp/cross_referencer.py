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
import json
import numpy as np
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