"""
Download MSMARCO-XI validation Parquet files from HuggingFace to local disk.
Validation files are ~440MB each — practical to download.
Train files are ~3.5GB each — too large for this environment.

We use validation split for BOTH:
  - indexing (building the Qdrant index with RECORDS_PER_LANGUAGE records)
  - retrieval evaluation (held-out subset using is_selected ground truth)

This is a documented limitation: index covers val split only.
Architecture scales to full train data when compute permits.

Forces IPv4 to avoid IPv6 connectivity issues on this machine.

Run from backend/:
    python scripts/download_data.py
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Force IPv4 before any network calls
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request
from app.config import settings
from app.utils.logger import configure_root_logger, get_logger

configure_root_logger(settings.log_level)
logger = get_logger("download_data")

REPO_BASE = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"
DATA_DIR = Path("data/msmarco_xi")

LANG_TO_VAL_PREFIX: Dict[str, str] = {
    "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
    "kn": "kan", "ml": "mal", "mr": "mar", "gu": "guj",
    "pa": "pan", "or": "ori", "as": "asm", "ne": "nep",
    "ur": "urd", "sa": "san",
}


def download_file_http(url: str, local_path: Path) -> bool:
    """Download file with progress reporting. Skip if already complete."""
    if local_path.exists():
        # Check size matches via HEAD
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=15) as r:
                remote_size = int(r.headers.get("Content-Length", 0))
            local_size = local_path.stat().st_size
            if local_size == remote_size:
                logger.info("Already complete",
                            extra={"path": str(local_path),
                                   "size_mb": round(local_size/1024/1024, 1)})
                return True
            else:
                logger.info("Incomplete, re-downloading",
                            extra={"local_mb": round(local_size/1024/1024,1),
                                   "remote_mb": round(remote_size/1024/1024,1)})
        except Exception:
            pass

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading", extra={"url": url, "dest": str(local_path)})

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python"})
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 1024  # 1MB chunks
            with local_path.open("wb") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = downloaded * 100 // total
                        if downloaded % (50 * 1024 * 1024) < chunk:  # log every 50MB
                            print(f"  {local_path.name}: {downloaded//1024//1024}MB / "
                                  f"{total//1024//1024}MB ({pct}%)", flush=True)
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info("Download complete",
                    extra={"path": str(local_path), "size_mb": round(size_mb, 1)})
        return True
    except Exception as e:
        logger.error("Download failed", extra={"url": url, "error": str(e)})
        if local_path.exists():
            local_path.unlink()
        return False


def main() -> None:
    languages = settings.language_list
    files_to_download: List[Tuple[str, Path]] = []

    for lang in languages:
        prefix = LANG_TO_VAL_PREFIX.get(lang)
        if not prefix:
            print(f"  Unknown language: {lang}")
            continue
        url = f"{REPO_BASE}/validation/{prefix}val.parquet"
        local = DATA_DIR / "validation" / f"{prefix}val.parquet"
        files_to_download.append((url, local))

    print(f"\nFiles to download ({len(files_to_download)}):")
    for url, local in files_to_download:
        print(f"  {local.name}")

    print("\nStarting downloads (validation files ~440MB each)...")
    success = failed = 0
    for url, local in files_to_download:
        ok = download_file_http(url, local)
        if ok:
            success += 1
        else:
            failed += 1

    total_bytes = sum(f.stat().st_size for f in DATA_DIR.rglob("*.parquet") if f.exists())
    print(f"\nDone: {success} OK, {failed} failed")
    print(f"Total on disk: {total_bytes/1024/1024:.0f} MB in {DATA_DIR}")


if __name__ == "__main__":
    main()
