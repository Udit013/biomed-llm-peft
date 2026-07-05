"""Document ingestion into normalized `Document`s.

Sources:
  * PubMed abstracts  — NCBI E-utilities (esearch + efetch), no API key required
                        (an email/api_key raises the rate limit). Reproducible from
                        a list of topic queries.
  * Guidelines        — NIH / WHO / CDC docs supplied as local text/markdown files
                        under corpus_dir/guidelines/<source>/*.txt (fetching PDFs is
                        out of scope; drop curated text there).

`iter_sample_documents()` yields a tiny in-memory corpus for tests/CI (no network).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from ..logging import get_logger
from ..schema import Document, Source

log = get_logger(__name__)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_pubmed(query: str, retmax: int = 200, email: str | None = None,
                 api_key: str | None = None) -> list[Document]:
    """esearch PMIDs for `query`, then efetch abstracts. Returns Documents."""
    import httpx

    common = {"db": "pubmed"}
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key

    with httpx.Client(timeout=30) as client:
        r = client.get(f"{EUTILS}/esearch.fcgi",
                       params={**common, "term": query, "retmax": retmax, "retmode": "json"})
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        time.sleep(0.34)  # be polite to NCBI (~3 req/s without key)
        r = client.get(f"{EUTILS}/efetch.fcgi",
                       params={**common, "id": ",".join(pmids), "retmode": "xml"})
        r.raise_for_status()
    return _parse_pubmed_xml(r.text)


def _parse_pubmed_xml(xml: str) -> list[Document]:
    docs: list[Document] = []
    root = ET.fromstring(xml)
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = (art.findtext(".//ArticleTitle") or "").strip()
        abstract = " ".join(
            (t.text or "").strip() for t in art.findall(".//Abstract/AbstractText")
        ).strip()
        if not abstract:
            continue
        year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
        journal = art.findtext(".//Journal/Title") or ""
        docs.append(Document(
            doc_id=f"pubmed:{pmid}", source="pubmed", title=title or "(untitled)",
            text=abstract, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            metadata={"year": _year(year), "journal": journal, "pmid": pmid},
        ))
    return docs


def load_guidelines(corpus_dir: str | Path) -> list[Document]:
    """Load curated NIH/WHO/CDC guideline text files from corpus_dir/guidelines/."""
    base = Path(corpus_dir) / "guidelines"
    docs: list[Document] = []
    for src in ("nih", "who", "cdc"):
        for path in sorted((base / src).glob("*.txt")) if (base / src).exists() else []:
            text = path.read_text(errors="ignore").strip()
            if text:
                docs.append(Document(
                    doc_id=f"{src}:{path.stem}", source=src,  # type: ignore[arg-type]
                    title=path.stem.replace("_", " "), text=text,
                    url=None, metadata={"file": str(path)}))
    return docs


def _year(raw: str) -> int:
    m = re.search(r"\d{4}", raw or "")
    return int(m.group()) if m else 0


def iter_sample_documents() -> Iterator[Document]:
    """A tiny offline corpus for tests/CI (no network, no models)."""
    samples = [
        ("pubmed:0001", "pubmed", "Metformin as first-line therapy in type 2 diabetes",
         "Metformin is recommended as the first-line pharmacologic treatment for type 2 "
         "diabetes mellitus. It lowers hepatic glucose production and improves insulin "
         "sensitivity, and is associated with a low risk of hypoglycemia.", 2019),
        ("who:hypertension_2021", "who", "WHO guideline on pharmacological treatment of hypertension",
         "The WHO recommends initiating antihypertensive treatment in adults with systolic "
         "blood pressure of 140 mmHg or higher. Thiazide diuretics, ACE inhibitors, and "
         "calcium channel blockers are recommended first-line agents.", 2021),
        ("cdc:vaccine_schedule", "cdc", "CDC adult immunization schedule",
         "The CDC recommends annual influenza vaccination for all adults. Adults aged 50 "
         "years and older should receive the recombinant zoster vaccine.", 2023),
    ]
    for doc_id, source, title, text, year in samples:
        yield Document(doc_id=doc_id, source=source, title=title, text=text,
                       metadata={"year": year})
