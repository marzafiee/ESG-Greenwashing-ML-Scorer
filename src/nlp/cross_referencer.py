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
import hashlib
import numpy as np
import uuid

from dotenv import load_dotenv
from datetime import date

load_dotenv()  # loading HF_TOKEN for models

# starter configurations
EMBED_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "roberta-large-mnli"
TOP_K = 10  # how many articles to check per claim
MIN_SIMILARITY = 0.3  # below this, an article isn't meaningful evidence for the claim
NLI_PARAGRAPHS = 8  # only feed the first N paragraphs of an article to the NLI model
OUTPUT_DIR = "data/processed/evidence"

# roberta-large-mnli returns these labels, we map them to our own verdict names as aforementioned
LABEL_TO_VERDICT = {
    "entailment": "SUPPORTS",
    "contradiction": "CONTRADICTS",
    "neutral": "UNVERIFIED",
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
        nli = pipeline("zero-shot-classification", model=NLI_MODEL)
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


# ~~~~~~~~ OTHER FUNCTIONS ~~~~~~~~~~~
def embed_articles(embedder, articles):
    """
    This function encodes every article's full_text once. Row i aligns to articles[i].
    """
    texts = [a["full_text"] for a in articles]
    return embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)


def top_matches(embedder, claim_text, article_embeddings, k=TOP_K):
    """
    This function returns [(article_index, similarity_score), ...] for the top k articles.
    """
    claim_emb = embedder.encode(claim_text, convert_to_tensor=True)
    scores = util.cos_sim(claim_emb, article_embeddings)[0]  # 1 row of scores
    k = min(k, len(scores))  # in case there are fewer than TOP_K articles total
    top_idx = np.argsort(scores.cpu().numpy())[::-1][:k]  # highest first
    return [(int(i), float(scores[i])) for i in top_idx]


def truncate_to_paragraphs(text, n=NLI_PARAGRAPHS):
    # keeping only the first n paragraphs before the NLI step. full articles dilute the signal (meaning lots of irrelevant context competing with the one sentence that actually matters), and this also keeps us well under the model's 512-token limit instead of relying on hard truncation mid-sentence.
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return text
    return "\n".join(paragraphs[:n])


def classify(nli, article_text, claim_text):
    """
    this function runs the NLI with the article as premise and the ESG claim as hypothesis.
    Uses zero-shot-classification pipeline with roberta-large-mnli.
    Returns (verdict, confidence) where confidence is the winning label's score.
    """
    premise = truncate_to_paragraphs(article_text)

    try:
        # zero-shot-classification takes the premise as the first argument and candidate_labels as the hypothesis options to score against.
        # roberta-large-mnli was trained on these three NLI labels specifically.
        result = nli(
            premise, candidate_labels=["entailment", "contradiction", "neutral"]
        )

        # zero-shot-classification returns a dict with "labels" and "scores" lists
        # already sorted highest score first so index 0 is always the winning label
        best_label = result["labels"][0]
        best_score = result["scores"][0]

        # then we map the NLI label to our project's verdict vocabulary
        verdict = LABEL_TO_VERDICT.get(best_label.lower(), "UNVERIFIED")
        return verdict, float(best_score)
    except Exception as e:
        # if NLI fails on one pair, don't kill the whole run — same pattern as the get_esg_category() function in claim_extractor.py, just mark it UNVERIFIED and move on
        print(f"Warning! NLI classification failed on one (claim, article) pair: {e}")
        return "UNVERIFIED", 0.0


# ~~~~~~~~~ saving ~~~~~~~~~~~~
def save_evidence_row(row, out_path, write_header):
    """
    This function appends a single evidence row to the csv immediately, so progress is never lost on a crash.
    """

    fieldnames = [
        "evidence_id",
        "claim_id",
        "company",
        "claim_text",
        "source_type",
        "source_name",
        "evidence_text",
        "verdict",
        "confidence_score",
        "similarity_score",
        "date_retrieved",
        "url",
    ]
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ~~~~~~~~~ MAIN MAIN FUNCTION (putting it all together) ~~~~~~~~~
def cross_reference(claims_csv_path, news_json_path, company_id="UNKNOWN"):
    """
    This function is the main entry point.
    Returns the list of evidence dicts it also saved.
    """

    claims = load_claims(claims_csv_path)
    articles = load_news(news_json_path)

    if not claims or not articles:
        print("No claims or no articles to process.")
        return []

    embedder, nli = load_models()
    article_embeddings = embed_articles(embedder, articles)  # encode once

    today = date.today().isoformat()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"evidence_{company_id}.csv")

    # checkpointing where if an evidence csv already exists for this company, load what's already there and skip re-processing
    # the same (claim, article) pairs on a re-run
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        evidence_rows = existing_rows
        already_processed = {
            hash_pair(r["claim_id"], r["url"] or r["source_name"])
            for r in existing_rows
        }
        write_header = False
        print(
            f"Resuming — found {len(evidence_rows)} existing evidence rows in checkpoint."
        )
    else:
        evidence_rows = []
        already_processed = set()
        write_header = True

    total_claims = len(claims)

    for i, claim in enumerate(claims):
        claim_text = claim.get("claim_text", "")
        claim_id = claim.get("claim_id", "")

        # skip malformed rows instead of letting the encoder choke on an empty string
        if not claim_text:
            print(
                f"[{i + 1}/{total_claims}] Skipping claim with no claim_text (claim_id={claim_id})."
            )
            continue

        print(f"[{i + 1}/{total_claims}] Checking: {claim_text[:60]}..")

        for idx, sim in top_matches(embedder, claim_text, article_embeddings):
            # skip near-irrelevant matches
            if sim < MIN_SIMILARITY:
                continue

            article = articles[idx]
            article_key = article.get("url") or article.get("title", "")

            # skip pairs we've already written on a previous run
            if hash_pair(claim_id, article_key) in already_processed:
                continue

            verdict, confidence = classify(nli, article["full_text"], claim_text)

            row = {
                "evidence_id": str(uuid.uuid4()),
                "claim_id": claim_id,
                "company": claim.get("company", ""),
                "claim_text": claim_text[:300],
                "source_type": "news",
                "source_name": article.get("source", article.get("title", "")),
                "evidence_text": article["full_text"][:500],  # snippet only
                "verdict": verdict,
                "confidence_score": round(confidence, 3),
                "similarity_score": round(sim, 3),
                "date_retrieved": today,
                "url": article.get("url", ""),
            }

            evidence_rows.append(row)
            save_evidence_row(row, out_path, write_header)
            write_header = False
            already_processed.add(hash_pair(claim_id, article_key))

            print(
                f"  ✓ Evidence saved ({len(evidence_rows)} total so far): verdict - {verdict}, sim - {round(sim, 3)}"
            )

    if not evidence_rows:
        print(
            "No evidence rows were generated. Check your claims/news inputs, or lower MIN_SIMILARITY."
        )
        return []

    print(f"\nDone! Saved {len(evidence_rows)} evidence rows to {out_path}")
    return evidence_rows


# quick test: this only runs when executing this file directly, not on import
if __name__ == "__main__":
    cross_reference(
        claims_csv_path="data/processed/claims/claims_Tesla, Inc._0001318605_0001628280-26-003952.csv",
        news_json_path="data/raw/news_tesla.json",
        company_id="TSLA",
    )
