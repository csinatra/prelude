"""Download raw corpus sources to data/raw/.

Usage: python -m ingest.download
Idempotent — skips files that already exist.
"""

import urllib.request
from pathlib import Path

from ingest.config import CODE4ML_FILES, LITE_COMPETITIONS, MLEBENCH_DESCRIPTION_URL, RAW_DIR


def _download(*, url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    for filename, url in CODE4ML_FILES.items():
        _download(url=url, dest=RAW_DIR / "code4ml" / filename)
    for slug in LITE_COMPETITIONS:
        _download(
            url=MLEBENCH_DESCRIPTION_URL.format(slug=slug),
            dest=RAW_DIR / "mlebench_descriptions" / f"{slug}.md",
        )
    print("done")


if __name__ == "__main__":
    main()
