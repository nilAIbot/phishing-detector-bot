import gzip
import re
from pathlib import Path

from bs4 import BeautifulSoup

EMAIL_RE = re.compile(r"\S+@\S+")
URL_RE   = re.compile(r"https?://\S+")
PHONE_RE = re.compile(r"\b\d{10}\b")


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return " ".join(text.split())


def process_nazario(gz_path: Path):
    with gzip.open(gz_path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("<html"):
                yield {
                    "text": clean_text(line),
                    "label": "phish",
                    "urls": URL_RE.findall(line),
                    "images": []
                }