### building an NLP inference pipeline that fetches SEC filings, cleans them, and uses a pretrained zero-shot classification model to extract and categorize ESG claims from the text.

import spacy  # NLP engine and sentence splitting
import os
import pandas as pd  # to build and save csvs
from transformers import pipeline  # Hugging Face which gives us ready-made AI model
from dotenv import load_dotenv
import re

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # runs if spaCy model hasn't been downloaded yet
    raise OSError("spaCy model not found! Run: python -m spacy download en_core_web_sm")

"""
Pipeline Architecture: 2-stage pipeline:
each sentence -> rule filter (is this a claim? -- high [recision filter]) -> ESG-BERT (what topic/category?) -> store structured claim
"""

# loading the Hugging Face zero-shot classifier
# updated from model: facebook/bart-large-mnli to ESG Specific model: nbroad/ESG-BERT
try:
    # classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    classifier = pipeline("text-classification", model="nbroad/ESG-BERT")
except Exception as e:
    # same as before - means transformers/torch aren't installed
    raise RuntimeError(f"Could not load ESG-BERT model: {e}")

# now for the labels we'd be asking our classifier to choose between. the model will score all of them -- for reference only
ESG_LABELS = ["Environmental", "Social", "Governance"]

# matches SEC section headers like "Item 1." or "Item 1A" for same pattern as preprocessor.py so that we can skip these parts when running
SECTION_HEADER_PATTERN = re.compile(r"^(item\s+\d+[a-zA-Z]?\.?)", re.IGNORECASE)


# layer 1: claim detection - rule-based, high precision
def is_esg_claim(sentence: str) -> bool:
    """
    determines whether a sentence is a verifiable ESG claim

    strict rule: For a sentence to be chosen as a claim, it must contain BOTH:
    1. numeric evidence OR measurable quantity
    2. action/change verb
    """
    text = sentence.lower()

    # numeric evidence
    has_numeric = bool(
        re.search(r"\d+%", text)
        or re.search(r"\d+\s*(tons|tonnes|kg|mt|co2|co₂)", text)
    )

    # action verbs checking
    action_verbs = [
        "reduced",
        "decreased",
        "increased",
        "improved",
        "achieved",
        "eliminated",
        "cut",
        "lowered",
        "offset",
        "prevented",
        "committed",
        "target",
        "aim",
        "expanded",
        "achieve",
    ]

    has_action = any(v in text for v in action_verbs)

    # exclusions i.e non-claims
    exclusions = [
        "headquarters",
        "incorporated",
        "board consists",
        "we operate in",
        "we are headquartered",
        "company was founded",
        "fiscal year",
        "shareholder meeting",
    ]

    if any(e in text for e in exclusions):
        return False

    return has_numeric and has_action


# layer 2: ESG topic classification
def get_esg_category(sentence: str) -> str:
    """
    Assigns ESG-BERT topic to validated claim sentences.
    For sentences that ARE ESG claims, this picks E, S, or G.
    """
    try:
        # using the ESG-BERT model for topic classification
        result = classifier(sentence)[0]
        return result["label"]
    except Exception as e:
        # if category tagging fails, we label it Unknown so the claim still gets saved rather than being silently lost.
        print(f"Warning! Category tagging failed :( : {e}")
        return "Unknown"


## our main pipeline
# input = cleaned text from preprocessor.py
def claim_extractor(cleaned_txt_file: str) -> list[dict]:
    """
    Takes a path to a .txt file (the cleaned text from preprocessor.py) and extracts claims from them
    """
    # reading the file given. but first we check if it exists
    if not os.path.exists(cleaned_txt_file):
        raise FileNotFoundError(f"Cleaned text file not found: {cleaned_txt_file}")

    # parsing the CIK num and accession number from the file name
    # filename format: cleaned_{company}_{CIK}_{accession_num}.txt
    filename = os.path.splitext(os.path.basename(cleaned_txt_file))[0]
    parts = filename.split("_", 3)  # 3 splits for ["cleaned", company, CIK, accession]

    if len(parts) != 4 or parts[0] != "cleaned":
        raise ValueError(
            f"Filename '{filename}' doesn't math the expected format: 'cleaned_{{company}}_{{CIK}}_{{accession_num}}.txt'"
        )

    company_name = parts[1]
    CIK_num = parts[2]
    accession_num = parts[3]

    print(
        f"Company: {company_name} | CIK_num: {CIK_num} | Accession No. : {accession_num}"
    )

    # LOAD TEXT
    # now we open the .txt file and read all the text into one large string
    try:
        with open(cleaned_txt_file, "r", encoding="utf-8") as f:
            cleaned_txt = f.read().strip()

        # just in case there's an error with the given .txt and it's actually blank
        if not cleaned_txt:
            raise ValueError(f"The file is empty!")

    except Exception as e:
        raise RuntimeError(f"Could not read file - {cleaned_txt_file}: {e}")

    # part 2! now to split the sentences with spaCy
    doc = nlp(
        cleaned_txt
    )  # passing our text through the language model to get back a Doc object. This object contains everything spaCy knows about our text — sentences, tokens, named entities, parts of speech. So like the "analysed" version of our text.

    sentences = list(
        doc.sents
    )  # doc.sents is a generator of all the sentences spaCy found. we'll place it in a list to see how many sentences there are.

    if not sentences:
        print("No sentences found in this doc. Check your input file again")
        return []

    # recent changes after first run: deduplicate sentences while preserving order
    # dict.fromkeys() uses the sentences as keys (keys are unique) then converts back to list
    sentences = list(dict.fromkeys(sent.text.strip() for sent in sentences))

    # the next part is to skip boilerplate before real content by finding the first "Item 1" header
    # next() walks the enumerated list and returns the index of the first match, or 0 if none found
    start_index = next(
        (i for i, s in enumerate(sentences) if SECTION_HEADER_PATTERN.match(s)),
        0,  # default to 0 (start of document) if no Item header is found
    )
    sentences = sentences[start_index:]  # slice everything before Item 1 away
    print(f"Starting from sentence {start_index} (first Item header found)")

    print(f"Found {len(sentences)} sentences. Running ESG classifier now.. ")

    # now adding CHECKPOINTING, i.e if a CSV already exists for this filing, load already-processed sentences
    # so we can skip them and resume from where we left off instead of starting over
    os.makedirs("data/processed/claims", exist_ok=True)
    outputPath = (
        f"data/processed/claims/claims_{company_name}_{CIK_num}_{accession_num}.csv"
    )

    if os.path.exists(outputPath):
        existing_df = pd.read_csv(outputPath)
        already_processed = set(existing_df["claim_text"].tolist())  # set = O(1) lookup
        claims = existing_df.to_dict(
            "records"
        )  # load existing claims back into our list
        print(f"Resuming — found {len(claims)} existing claims in checkpoint.")
    else:
        already_processed = set()
        claims = []

    # main loop
    for i, sentence_text in enumerate(sentences):
        # skipping blank or very short sentences like page nums, etc
        if len(sentence_text) < 20:
            continue

        # skip section headers and part markers that aren't real sentences
        if re.match(
            r"^(PART\s+[IVX]+|Item\s+\d+|Mine Safety)",
            sentence_text.strip(),
            re.IGNORECASE,
        ):
            continue

        # skip sentences we already processed in a previous run
        if sentence_text in already_processed:
            print(f"[{i + 1} / {len(sentences)}] Skipping (already processed)..")
            continue

        # checking for progress!
        # indicator: zero-shot classification is slow (approx. 1-2s per sentence) so we'll know if it's working
        print(f"[{i + 1} / {len(sentences)}] Checking: {sentence_text[:60]}..")

        # layer 1: claim filter
        # for each sentence, we run zero-shot classifier: is this an ESG claim? yes/no
        if not is_esg_claim(sentence_text):
            continue

        # TAGGING CATEGORIES -- ESG classification
        category = get_esg_category(sentence_text)

        # one dict = one row in the final CSV
        claims.append(
            {
                "company": company_name,
                "CIK": CIK_num,
                "Accession No.": accession_num,
                "category": category,  # E/S/G
                "claim_text": sentence_text,  # actual sentence claim
            }
        )

        # FIX 3 — write each new claim immediately so progress is never lost on a crash
        # mode='a' appends one row, header=False skips rewriting column names each time
        pd.DataFrame([claims[-1]]).to_csv(
            outputPath,
            mode="a",
            header=not os.path.exists(outputPath) or i == 0,
            index=False,
        )
        print(f"  ✓ Claim saved ({len(claims)} total so far)")

    # WARNING if nothing was found. problem is most likely from the model or an input issue
    if not claims:
        print("No ESG claims found. The file may not contain ESG-related content")
        return []

    print(f"\nDone! Saved {len(claims)} ESG claims for {company_name} to {outputPath}")

    # return list of claim dicts so that our pipeline script can use the claims directly without rereading the CSV.
    return claims


# quick test
results = claim_extractor(
    "data/cleaned/cleaned_Tesla, Inc._0001318605_0001628280-26-003952.txt"
)
print(f"\nReturned {len(results)} claims.")
