# shows every sentence that has ESG terms
import re
import spacy

nlp = spacy.load("en_core_web_sm")

with open(
    "data/cleaned/cleaned_Tesla, Inc._0001318605_0001628280-26-003952.txt",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()

doc = nlp(text)
sentences = list(dict.fromkeys(sent.text.strip() for sent in doc.sents))

esg_terms = [
    "emission",
    "emissions",
    "carbon",
    "ghg",
    "supercharger",
    "charging",
    "electric vehicle",
    "clean energy",
    "grid",
    "storage",
    "gigafactory",
    "climate",
    "energy",
    "renewable",
    "solar",
    "battery",
    "safety",
    "injury",
    "workplace",
    "labor",
    "diversity",
    "inclusion",
    "human rights",
    "privacy",
    "data security",
    "cybersecurity",
    "sustainability",
    "waste",
    "water",
    "net zero",
]

for sent in sentences:
    if any(term in sent.lower() for term in esg_terms):
        print(sent[:150])
        print("---")
