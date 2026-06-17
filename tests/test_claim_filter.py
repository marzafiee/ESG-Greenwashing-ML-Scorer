# tests for layer 1 of the claim pipeline — the rule-based is_esg_claim() pre-filter
# imports the real function from claim_extractor so we don't maintain a duplicate copy

import sys
from pathlib import Path
from unittest.mock import MagicMock

# mock transformers before importing claim_extractor — this test only needs is_esg_claim(), not ESG-BERT
sys.modules["transformers"] = MagicMock()

# add src/ to path so we can import nlp.claim_extractor like the rest of the pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp.claim_extractor import is_esg_claim  # noqa: E402 — path must be set first


# real ESG claims we expect the filter to keep — mix of outcome, operational, and commitment sentences
SHOULD_CAPTURE = [
    "In 2025, we deployed 46.7 GWh of energy storage products.",
    "Where possible, we co-locate Superchargers with our solar and energy storage systems to reduce costs and promote renewable power.",
    "We have committed to achieving net zero emissions by 2040.",
    "We reduced our manufacturing waste by 30% in 2025.",
    "Our Gigafactories are powered by renewable energy.",
    "Tesla installed 10,000 solar panels at our Austin facility.",
    "We expanded our Supercharger network to 60,000 stations globally.",
    "Our vehicles avoided 20 million metric tons of CO2 emissions.",
]

# financial, regulatory, risk, and boilerplate sentences the filter should drop
SHOULD_REJECT = [
    "Gross margin for total automotive decreased from 18.4% to 17.8%",
    "These incentives may expire when the allocated funding is exhausted",
    "We are required to comply with other federal laws administered by NHTSA",
    "10-Q 001-34756 10.2 July 29, 2019",
]


def test_captures_esg_claims():
    # every sentence in SHOULD_CAPTURE must pass the pre-filter
    for sentence in SHOULD_CAPTURE:
        assert is_esg_claim(sentence), f"Expected claim but got reject: {sentence}"


def test_rejects_non_claims():
    # every sentence in SHOULD_REJECT must be blocked before ESG-BERT runs
    for sentence in SHOULD_REJECT:
        assert not is_esg_claim(sentence), f"Expected reject but got claim: {sentence}"


if __name__ == "__main__":
    test_captures_esg_claims()
    test_rejects_non_claims()
    print(f"All {len(SHOULD_CAPTURE) + len(SHOULD_REJECT)} claim-filter tests passed.")
