"""Content extraction heuristics for converted Markdown documents."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List

_COLOR_KEYWORDS = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "gold",
    "silver",
    "beige",
    "yellow",
    "purple",
    "pink",
    "gray",
}

_MATERIAL_KEYWORDS = {
    "cotton",
    "silk",
    "leather",
    "denim",
    "wool",
    "linen",
    "wood",
    "plastic",
    "metal",
    "glass",
}

_PRODUCT_TYPES = {
    "dress",
    "shoe",
    "bag",
    "coat",
    "jacket",
    "shirt",
    "pants",
    "skirt",
    "scarf",
    "accessory",
}

_PRICE_PATTERN = re.compile(r"\$[\d,]+(?:\.\d+)?")
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")


@dataclass
class ExtractionResult:
    """Structured metadata extracted from a document."""

    concepts: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    prices: List[str] = field(default_factory=list)
    product_types: List[str] = field(default_factory=list)
    clients: List[str] = field(default_factory=list)
    product_lines: List[str] = field(default_factory=list)
    proper_nouns: List[str] = field(default_factory=list)

    def asdict(self) -> dict:
        return {
            "concepts": self.concepts,
            "colors": self.colors,
            "materials": self.materials,
            "prices": self.prices,
            "product_types": self.product_types,
            "clients": self.clients,
            "product_lines": self.product_lines,
            "proper_nouns": self.proper_nouns,
        }


def _load_spacy_model():  # pragma: no cover - optional dependency
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


_NLP = _load_spacy_model()


def _extract_proper_nouns(text: str) -> List[str]:
    if _NLP is None:
        # Simple fallback: capitalised words that are not sentence starters.
        tokens = _WORD_PATTERN.findall(text)
        proper = [tok for tok in tokens if tok[0].isupper() and not tok.isupper()]
        return sorted(set(proper))

    doc = _NLP(text)
    return [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT"}]


def _classify_clients_and_products(names: Iterable[str]) -> tuple[List[str], List[str]]:
    clients: List[str] = []
    product_lines: List[str] = []
    for name in names:
        normalized = name.lower()
        if any(suffix in normalized for suffix in (" inc", " ltd", " llc", " corporation")):
            clients.append(name)
        else:
            product_lines.append(name)
    return sorted(set(clients)), sorted(set(product_lines))


def extract_entities(markdown_text: str) -> ExtractionResult:
    """Extract structured metadata from Markdown text."""

    # Remove Markdown syntax for lighter parsing
    clean_text = re.sub(r"[#>*_`]+", " ", markdown_text)

    prices = _PRICE_PATTERN.findall(clean_text)

    tokens = [token.lower() for token in _WORD_PATTERN.findall(clean_text)]
    colors = sorted({token for token in tokens if token in _COLOR_KEYWORDS})
    materials = sorted({token for token in tokens if token in _MATERIAL_KEYWORDS})
    product_types = sorted({token for token in tokens if token in _PRODUCT_TYPES})

    proper_nouns = _extract_proper_nouns(clean_text)
    clients, product_lines = _classify_clients_and_products(proper_nouns)

    # Candidate "concepts" are the most common nouns / meaningful tokens.
    # With spaCy we can use part-of-speech tags; otherwise frequency heuristics.
    if _NLP is None:
        counter = Counter(token for token in tokens if len(token) > 3)
        concepts = [word for word, _ in counter.most_common(15)]
    else:
        doc = _NLP(clean_text)
        counter = Counter(token.lemma_.lower() for token in doc if token.pos_ == "NOUN")
        concepts = [word for word, _ in counter.most_common(20)]

    return ExtractionResult(
        concepts=concepts,
        colors=colors,
        materials=materials,
        prices=prices,
        product_types=product_types,
        clients=clients,
        product_lines=product_lines,
        proper_nouns=sorted(set(proper_nouns)),
    )
