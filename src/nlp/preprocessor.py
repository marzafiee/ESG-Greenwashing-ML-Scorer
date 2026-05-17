# cleans raw HTML from filing_fetcher.py down to readable, structured text before any NLP can happen.
# unlike the simple preprocessor, this handles iXBRL tags, hidden elements, bullet reconstruction, section header preservation, and deduplication of SEC boilerplate patterns.

import re
import html
import unicodedata
import os

from bs4 import BeautifulSoup, Comment  # parses HTML into a navigable tree we can clean
from collections import (
    Counter,
)  # counts line frequency so we can drop repeated boilerplate stuff

from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# our fetched documents are visually rendered SEC filings, not semantic HTML.
# they were designed for browser rendering and printing, not NLP.
# so formatting tags alone are meaningless meaning we have to reconstruct paragraphs, bullets,
# and section headers from the visual layout patterns ourselves.


class SECFilingCleaner:
    """
    Regex Patterns: these are compiled once at class level for efficiency and reused across every call to clean()
    """

    PAGE_NUMBER_PATTERN = re.compile(
        r"^\d+$"
    )  # matches lines that are ONLY a number, e.g. "47" — these are page numbers

    TABLE_OF_CONTENTS_PATTERN = re.compile(
        r"table\s+of\s+contents",  # SEC filings are known to repeat the TOC on nearly every page
        re.IGNORECASE,
    )

    MULTISPACE_PATTERN = re.compile(
        r"[ \t]+"
    )  # matches runs of spaces or tabs and will be collapsed to single space

    MULTINEWLINE_PATTERN = re.compile(
        r"\n{3,}"
    )  # 3 or more consecutive newlines and will be collapsed to 2 (one blank line)

    HTML_ENTITY_PATTERN = re.compile(
        r"&#\d+;"
    )  # matches numeric HTML entities like &#8217; — handled by html.unescape() in step 1

    SECTION_HEADER_PATTERN = re.compile(
        r"^(item\s+\d+[a-zA-Z]?\.?)", re.IGNORECASE
    )  #  matches SEC section headers like "Item 1." or "Item 1A"

    REPEATED_JUNK_PATTERN = re.compile(
        r"^(table of contents|\d+)$", re.IGNORECASE
    )  # catches standalone page numbers and TOC lines

    BULLET_PATTERN = re.compile(
        r"^[•·▪■\-]|\u2022"
    )  # matches common bullet characters — \u2022 is the unicode bullet •

    def __init__(
        self,
        preserve_bullets=True,  # if True, bullet lines get normalized to "- " prefix
        preserve_headers=True,  # if True, "Item X" headers get blank lines around them
        deduplicate_lines=True,  # if True, lines repeated more than 5 times are dropped
    ):
        self.preserve_bullets = preserve_bullets
        self.preserve_headers = preserve_headers
        self.deduplicate_lines = deduplicate_lines

    """
    main pipeline: this is the only public method called by the way
    """

    def clean(self, raw_html: str) -> str:
        """
        Takes raw HTML from filing_fetcher.py and returns cleaned plain text
        ready for NLP processing. Handles iXBRL tags, hidden elements, HTML entities,
        page artifacts, bullet normalization, and SEC section header preservation.
        """
        # decode HTML entities before parsing so BS4 sees clean text
        raw_html = self._decode_html_entities(raw_html)

        # then we parse into a navigable tree using lxml (as it's faster + handles malformed XHTML)
        soup = BeautifulSoup(raw_html, "lxml")

        # strip HTML comments (often contain hidden XBRL metadata blocks)
        self._remove_comments(soup)

        # remove display:none elements (iXBRL duplicate data blocks invisible to browsers)
        self._remove_hidden_elements(soup)

        # decompose XBRL namespace tags like <ix:nonNumeric>, <us-gaap:Revenue>
        self._remove_xbrl_metadata(soup)

        # remove scripts, styles, images, and fake PDF page-break <hr> tags
        self._remove_layout_noise(soup)

        # extract text block by block (not all at once) to preserve line boundaries
        text = self._extract_semantic_text(soup)

        # NFKC normalization: converts ligatures, full-width chars, non-breaking spaces
        text = self._normalize_unicode(text)

        # split into a list of lines so we can filter them individually
        lines = self._split_lines(text)

        # drop page numbers, TOC lines, ultra-short noise (< 3 chars)
        lines = self._clean_lines(lines)

        # drop lines repeated more than 5 times (company name in headers, footer text etc.)
        if self.deduplicate_lines:
            lines = self._deduplicate_lines(lines)

        # add blank lines around Item headers, normalize bullet characters to "- "
        text = self._rebuild_document(lines)

        # collapse runs of spaces/tabs and runs of 3+ newlines
        text = self._normalize_whitespace(text)

        return (
            text.strip()
        )  # remove any leading/trailing whitespace from the final result

    def _decode_html_entities(self, text):
        # must run on the raw string BEFORE BeautifulSoup parses it,
        # otherwise entities get baked into text nodes e.g. &#8217; → ' and &#8226; → •
        return html.unescape(text)

    def _remove_comments(self, soup):
        # SEC iXBRL documents often embed XBRL metadata inside <!-- --> comment blocks
        # invisible in a browser but get_text() would still extract them
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()  # .extract() removes the node but keeps the rest of the tree intact

    def _remove_hidden_elements(self, soup):
        # iXBRL hides duplicate financial data blocks using style="display:none"
        # .decompose() removes the element AND all its children so get_text() never sees them
        hidden = soup.find_all(
            style=lambda value: value and "display:none" in value.lower()
        )
        for element in hidden:
            element.decompose()

    def _remove_xbrl_metadata(self, soup):
        # iXBRL wraps numbers in machine-readable tags like <ix:nonNumeric> and <us-gaap:Revenue>
        # these are annotations for the SEC's XBRL parser so not human-readable text
        xbrl_prefixes = [
            "ix:",  # inline XBRL viewer tags
            "xbrli:",  # XBRL instance document tags
            "dei:",  # document and entity information tags
            "us-gaap:",  # US GAAP financial taxonomy tags
        ]
        for tag in soup.find_all():
            if any(tag.name.startswith(prefix) for prefix in xbrl_prefixes):
                tag.decompose()

    def _remove_layout_noise(self, soup):
        # removes tags with zero textual value: scripts, styles, images, and embedded frames
        # <hr> tags in SEC filings are fake PDF page breaks (style="page-break-after:always") are just pure noise
        unwanted = [
            "script",  # javascript
            "style",  # css
            "meta",  # document metadata
            "link",  # external resource references
            "iframe",  # embedded frames
            "svg",  # vector graphics
            "img",  # raster images
        ]
        for tag in soup(unwanted):
            tag.decompose()
        for hr in soup.find_all("hr"):
            hr.decompose()  # fake PDF page-break markers, not structural dividers

    def _extract_semantic_text(self, soup):
        # iterates block by block instead of soup.get_text() on the whole doc (which smashes everything together)
        # separator=" " prevents "revenueincreased" type concatenation when inline tags are stripped
        # only extract from elements that have no block-level children
        # this prevents parent divs and their child spans from both being extracted (causes duplication)
        BLOCK_TAGS = {"div", "p", "li"}
        blocks = []
        for element in soup.find_all(["div", "p", "li", "span"]):
            has_block_child = any(
                child.name in BLOCK_TAGS
                for child in element.children
                if hasattr(child, "name")
            )
            if has_block_child:
                continue  # skip — its children will be extracted individually
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue  # skip empty elements
            blocks.append(text)
        return "\n".join(blocks)  # each block on its own line for _split_lines

    def _normalize_unicode(self, text):
        # NFKC converts lookalike characters to standard forms: "fi" ligature to "fi",
        # full-width digits to normal digits, non-breaking spaces (\xa0) to regular spaces
        return unicodedata.normalize("NFKC", text)

    def _split_lines(self, text):
        # splits on newlines from _extract_semantic_text and strips each line
        # gives us a list of strings to filter individually in _clean_lines
        return [line.strip() for line in text.split("\n")]

    def _clean_lines(self, lines):
        # four filters per line: drops empty, standalone page numbers, TOC repetitions, and sub-3-char noise
        # anything surviving all four is kept
        cleaned = []
        for line in lines:
            if not line:
                continue  # skip blank lines
            if self.PAGE_NUMBER_PATTERN.match(line):
                continue  # skip standalone page numbers e.g. "47"
            if self.TABLE_OF_CONTENTS_PATTERN.search(line):
                continue  # skip repeated TOC header lines
            if self.REPEATED_JUNK_PATTERN.match(line):
                continue  # skip lines that are only a number or only "table of contents"
            if len(line) < 3:
                continue  # skip ultra-short noise like ".", "-", "—"
            cleaned.append(line)
        return cleaned

    def _deduplicate_lines(self, lines):
        # Counter counts occurrences of each unique line across the whole document
        # threshold of 5 allows legitimate repetition so anything above is boilerplate (headers, footers)
        counts = Counter(
            lines
        )  # e.g. {"Tesla, Inc.": 42, "Item 1A. Risk Factors": 2, ...}
        cleaned = []
        for line in lines:
            if counts[line] > 5:
                continue  # drop boilerplate repeated more than 5 times
            cleaned.append(line)
        return cleaned

    def _rebuild_document(self, lines):
        # SEC Item headers get blank lines before and after so chunkers and RAG systems can split on them
        # bullet characters (•, ▪, ■) get normalized to "- " so downstream NLP treats them uniformly
        rebuilt = []
        for line in lines:
            if self.preserve_headers and self.SECTION_HEADER_PATTERN.match(line):
                rebuilt.append(
                    f"\n{line}\n"
                )  # blank line before and after Item headers
                continue
            if self.preserve_bullets and self.BULLET_PATTERN.search(line):
                rebuilt.append(f"- {line}")  # normalize all bullet styles to "- "
                continue
            rebuilt.append(line)  # plain paragraph line, no transformation needed
        return "\n".join(rebuilt)

    def _normalize_whitespace(self, text):
        # two final passes: collapse space/tab runs to one space, collapse 3+ newlines to 2
        text = self.MULTISPACE_PATTERN.sub(" ", text)  # collapse spaces/tabs
        text = self.MULTINEWLINE_PATTERN.sub("\n\n", text)  # collapse 3+ newlines to 2
        return text


# ran directly to test against the Tesla 10-K fetched by filing_fetcher.py
if __name__ == "__main__":
    from filing_fetcher import filing_fetcher

    # fetch the raw HTML from SEC EDGAR
    raw_html = filing_fetcher(
        "sec_Tesla, Inc.(0001318605)_10k_filings.json", "0001628280-26-003952"
    )

    cleaner = SECFilingCleaner()
    clean_text = cleaner.clean(raw_html)

    # preview first 3000 chars
    print(clean_text[:3000])

    # save cleaned output
    os.makedirs("data/cleaned", exist_ok=True)
    with open("data/cleaned/tesla_10k_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"\nTotal characters after cleaning: {len(clean_text):,}")
