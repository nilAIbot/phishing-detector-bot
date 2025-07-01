"""Download & archive phishing + benign corpora (open‑source feeds)."""

import os, requests, datetime as dt, logging
from pathlib import Path

RAW_DIR = Path(os.getenv("RAW_DIR", "./../../data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)

# Sources
URLHAUS_TXT = "https://urlhaus.abuse.ch/downloads/text_recent"
ENRON_URL   = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"

def _safe_download(url: str, out_path: Path) -> bool:
    """Download URL to a local path safely, with error logging."""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        log.info("✔ Downloaded %s → %s", url.split("/")[-1], out_path)
        return True
    except requests.exceptions.HTTPError as e:
        log.warning("✖ HTTP error: %s — %s", url, e)
    except requests.exceptions.RequestException as e:
        log.warning("✖ Network error: %s — %s", url, e)
    return False

def update():
    """Download latest phishing and benign data into raw/."""
    today = dt.date.today().isoformat()

    # 1. URLHaus
    urlhaus_dst = RAW_DIR / f"urlhaus_{today}.txt"
    if not urlhaus_dst.exists():
        _safe_download(URLHAUS_TXT, urlhaus_dst)

    # 2. Enron (once)
    enron_dst = RAW_DIR / "enron_mail.tar.gz"
    if not enron_dst.exists():
        _safe_download(ENRON_URL, enron_dst)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    update()
