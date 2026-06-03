import re


def is_esg_claim(sentence: str) -> bool:
    """
    determines whether a sentence is a verifiable ESG claim

    strict rule: For a sentence to be chosen as a claim, it:
    - MUST contain either action OR numeric OR commitment (numeric evidence OR measurable quantity OR action/change verb)
    - should be ESG context
    - BUT must NOT be purely regulatory language
    """
    text = sentence.lower()

    # reject obvious non-claims
    reject_patterns = [
        "we are required to",
        "must comply",
        "subject to",
        "risk",
        "uncertainty",
        "could result in",
        "may result in",
        "may adversely affect",
        "cannot assure",
        "tax credit",
        "tax credits",
        "internal revenue code",
        "gross margin",
        "revenue increased",
        "revenue decreased",
        "operating income",
        "net income",
        "cash flow",
        "financial results",
        "selling price",
        "cost per unit",
        "10-q",
        "10-k",
        "exhibit",
        "certification",
        "rule 13a",
    ]

    if any(p in text for p in reject_patterns):
        return False

    # esg context
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

    has_esg = any(term in text for term in esg_terms)

    if not has_esg:
        return False

    # numeric evidence
    has_numeric = bool(
        re.search(r"\d+%", text)
        or re.search(
            r"\d+\.?\d*\s*(million|billion|tons|tonnes|kg|mt|co2|co₂|CO2|gwh|mwh|kwh|twh|gj|mj)",
            text,
        )
        or re.search(r"\d{1,3}(,\d{3})+", text)  # catches 10,000 / 60,000 / 1,000,000
    )

    # action verbs checking
    action_verbs = [
        "reduced",
        "decreased",
        "increased",
        "avoided",
        "improved",
        "achieved",
        "eliminated",
        "cut",
        "lowered",
        "offset",
        "prevented",
        "expanded",
        "developed",
        "deployed",
        "launched",
        "implemented",
        "powered",
        "installed",
        "avoided",
        "sourced",
        "diverted",
    ]

    has_action = any(v in text for v in action_verbs)

    # updated to capture commitment phrases
    commitment_phrases = [
        "committed to",
        "net zero",
        "by 2030",
        "by 2040",
        "by 2050",
        "target of",
        "goal of",
        "we aim to",
        "plan to achieve",
        "carbon neutral",
        "100% renewable",
        "powered by renewable",
        "100% renewable",
        "powered by clean",
    ]
    has_commitment = any(p in text for p in commitment_phrases)

    return (has_numeric and has_action) or has_commitment


sentences = [
    "We reduced our manufacturing waste by 30% in 2025.",
    "Our Gigafactories are powered by renewable energy.",
    "We have committed to achieving net zero emissions by 2040.",
    "Tesla installed 10,000 solar panels at our Austin facility.",
    "We expanded our Supercharger network to 60,000 stations globally.",
    "Our vehicles avoided 20 million metric tons of CO2 emissions.",
]

for s in sentences:
    print(is_esg_claim(s), "|", s[:60])
