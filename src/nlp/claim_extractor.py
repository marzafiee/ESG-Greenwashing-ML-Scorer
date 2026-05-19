# Split the cleaned text into individual sentences  with spaCy.
import spacy  # NLP engine and sentence splitting
import os
import pandas as pd  # to build and save csvs
from transformers import pipeline  # Hugging Face which gives us ready-made AI model
from dotenv import load_dotenv

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # runs if spaCy model hasn't been downloaded yet
    raise OSError("spaCy model not found! Run: python -m spacy download en_core_web_sm")

# loading the Hugging Face zero-shot classifier
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    # same as before - means transformers/torch aren't installed
    raise RuntimeError(f"Could not load Hugging Faceclassifier: {e}")

# now for the labels we'd be asking our classifier to choose between. the model will score all of them
ESG_LABELS = ["Environmental", "Social", "Governance"]


def is_esg_claim(sentence: str) -> bool:
    """
    then to tag a category, we ask a yes / no for: "is this sentence an ESG claim at all?"
    then we run the heavier category tagger on sentences that pass
    """
    try:
        # giving the classifier 2 candidate labels
        result = classifier(
            sentence, candidate_labels=["ESG claim", "NOT an ESG claim"]
        )

        # result["labels"]  -  sorts by score with highest first
        # thus, result["labels"][0] gives the winning label
        return result["labels"][0] == "ESG claim"
    except Exception as e:
        # if the classifier happens to fail on a sentence, we will log it and skip instead of craching the model run
        print(f"Warning! ESG check failed for sentence: {e}")
        return False


def get_esg_category(sentence: str) -> str:
    """
    For sentences that ARE ESG claims, this picks E, S, or G.
    """
    try:
        # using the same zero-shot approach but now with 3 categories
        result = classifier(sentence, candidate_labels=ESG_LABELS)
        return result["labels"][0]
    except Exception as e:
        # if category tagging fails, we label it Unknown so the claim still gets saved rather than being silently lost.
        print(f"Warning! Category tagging failed :( : {e}")
        return "Unknown"


## our main function!
# input = cleaned text from preprocessor.py
def claim_extractor(cleaned_txt_file: str) -> list[dict]:
    """
    Takes a path to a .txt file (the cleaned text from preprocessor.py) and extracts claims from them
    """
    # reding the file given. but first we check if it exists
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

    # now we open the .txt file and read all the test into onr large string
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
        print("No sentences founf in this doc. Check your input file again")
        return []

    print(f"Found {len(sentences)} sentences. Running ESG classifier now.. ")

    # now we'll only collect the ESG claims in this list
    claims = []

    for i, sent in enumerate(sentences):
        # sent.text : plain str of sentences i.e no spaCy metadata
        sentence_text = sent.text.strip()

        # skipping blank or very short sentences like page nums, etc
        if len(sentence_text) < 10:
            continue

        # checking for progress!
        # indicator: zero-shot classification is slow (approx. 1-2s per sentence) so we'll know if it's working
        print(f"[{i + 1} / {len(sentences)}] Checking: {sentence_text[:60]}..")

        # for each sentence, we run zero-shot classifier: is this an ESG claim? yes/no
        # doc.sents into  individual sentences
        if is_esg_claim(sentence_text):
            # is_esg_claim() returns True or False

            # TAGGING CATEGORIES
            category = get_esg_category(sentence_text)

            # one dict = one row in the final CSV
            claims.append(
                {
                    "CIK": CIK_num,
                    "Accession No.": accession_num,
                    "category": category,  # if E/S/G
                    "claim_text": sentence_text,  # actual sentence claim
                }
            )

    # WARNING if nothing was found. problem is most likely from the model or an input issue
    if not claims:
        print("No ESG claims found. The file may not contain ESG-related content")
        return []

    # now to save to .CSV
    # converting the list of dictionaries into a pd dataframe (df)
    # df == table where rows: claims and cols = the dict keys
    df = pd.DataFrame(claims)

    # save to data/processed/claims_{CIK}_{accession}.csv
    # making sure it exists
    os.makedirs("data/processed/claims", exist_ok=True)

    # save to csv file
    outputPath = (
        f"data/processed/claims/claims_{company_name}_{(CIK_num)}_{accession_num}.csv"
    )

    # writing the df to CSV. Index=False means do not write
    try:
        df.to_csv(outputPath, index=False)
    except Exception as e:
        raise RuntimeError(f"Could not save CSV TO {outputPath}: {e}")

    print(f"\nDone! Saved {len(claims)} ESG claims for {company_name} to {outputPath}")

    # return list of claim dicts so that our pipeline script can use the claims directly without rereading the CSV.
    return claims


# quick test
results = claim_extractor(
    "data/cleaned/cleaned_Tesla, Inc._0001318605_0001628280-26-003952.txt"
)
print(f"\nReturned {len(results)} claims.")
