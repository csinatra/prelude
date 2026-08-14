"""Shared constants for corpus download and ingestion.

Corpus sources (dev subset — see implementation brief):
- Code4ML (zenodo 6607065): competitions.csv → competition_metadata collection;
  code_blocks CSVs (already cell-level chunks, tagged with competition slug and
  Kaggle score) → practitioner_knowledge collection.
- mle-bench repo: description.md per competition → competition_metadata.

The brief's Source 1 (MLEModernizer, zenodo 15022707) ships as a single 107 GB
tar.gz and is deferred to the cloud box — not downloadable piecemeal.
"""

from pathlib import Path

RAW_DIR = Path("data/raw")
CHROMA_PATH = Path("data/chroma")

COMPETITION_METADATA = "competition_metadata"
PRACTITIONER_KNOWLEDGE = "practitioner_knowledge"
NOTEBOOK_SUMMARIES = "notebook_summaries"

CODE4ML_RECORD = "https://zenodo.org/api/records/6607065/files"
# Note: the "с" in the code_blocks filenames on Zenodo is Cyrillic U+0441.
CODE4ML_FILES = {
    "competitions.csv": f"{CODE4ML_RECORD}/competitions.csv/content",
    "code_blocks_upto_20.csv": f"{CODE4ML_RECORD}/%D1%81ode_blocks_upto_20.csv/content",
    "code_blocks_21.csv": f"{CODE4ML_RECORD}/%D1%81ode_blocks_21.csv/content",
}

MLEBENCH_DESCRIPTION_URL = (
    "https://raw.githubusercontent.com/openai/mle-bench/main/mlebench/competitions/{slug}/description.md"
)

# Notebook-selection scopes for the practitioner corpus (ingest.ingest_summaries).
#   "lite" — notebooks from the Lite-22 competitions only. The dev corpus: small
#            enough to rebuild cheaply while iterating on prompts or embeddings.
#   "full" — every competition in Code4ML. Leave-one-out at query time still
#            excludes the competition being specified, so including the eval
#            competitions' own notebooks is not leakage.
# CORPUS_SCOPES maps a scope to the competition allowlist, or None for no filter.
DEFAULT_SCOPE = "lite"

# MLE-bench Lite — the 22 Low-complexity competitions (experiments/splits/low.txt).
LITE_COMPETITIONS = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]

CORPUS_SCOPES: dict[str, list[str] | None] = {
    "lite": LITE_COMPETITIONS,
    "full": None,
}
