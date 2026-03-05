"""
KSelect RAG demo — Titanic dataset stress test.

Converts each Titanic passenger row into a natural-language narrative,
indexes them with KSelect, then fires hard factual + analytical questions
to see how well retrieval + Claude handle structured tabular data.

Setup:
    pip install kagglehub[pandas-datasets] anthropic python-dotenv pandas
    PYTHONPATH=. python main2.py
"""
from __future__ import annotations

import os
import tempfile
import csv
import time
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"


def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[timer] {fn.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

_PCLASS_MAP = {1: "first", 2: "second", 3: "third"}
_EMBARKED_MAP = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
_METADATA_COLS = ["PassengerId", "Survived", "Pclass", "Name", "Sex",
                  "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"]


def _is_missing(val) -> bool:
    return val is None or str(val).strip().lower() in ("", "nan")


# ── 1. Load Titanic from Kaggle ───────────────────────────────────────────────
def load_titanic_df():
    import kagglehub
    import pandas as pd

    print("[kaggle] Downloading Titanic dataset...")
    dataset_path = kagglehub.dataset_download("yasserh/titanic-dataset")
    csv_files = list(Path(dataset_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {dataset_path}")
    df = pd.read_csv(csv_files[0])
    print(f"[kaggle] Loaded {len(df)} rows | {csv_files[0].name}")
    return df


# ── 2. Convert a passenger row → natural-language narrative ──────────────────
def row_to_narrative(row: dict) -> str:
    survived = "survived" if row.get("Survived") == 1 else "did not survive"
    pclass = _PCLASS_MAP.get(int(row.get("Pclass", 0)), str(row.get("Pclass", "unknown")))
    name = row.get("Name", "Unknown")
    sex = row.get("Sex", "unknown")
    age = row.get("Age")
    fare = row.get("Fare")
    cabin = row.get("Cabin")
    ticket = row.get("Ticket", "unknown")
    embarked = _EMBARKED_MAP.get(str(row.get("Embarked", "")), "an unknown port")

    age_str = f"{float(age):.0f} years old" if not _is_missing(age) else "of unknown age"
    fare_str = f"£{float(fare):.2f}" if not _is_missing(fare) else "unknown fare"
    cabin_str = f"cabin {cabin}" if not _is_missing(cabin) else "no recorded cabin"

    sibsp = int(row.get("SibSp", 0))
    parch = int(row.get("Parch", 0))
    family = []
    if sibsp > 0:
        family.append(f"{sibsp} sibling(s) or spouse(s)")
    if parch > 0:
        family.append(f"{parch} parent(s) or child(ren)")
    family_str = " and ".join(family) if family else "no family members"

    return (
        f"Passenger {name} (ticket {ticket}) was a {age_str} {sex} traveling in {pclass} class "
        f"who {survived} the Titanic disaster. "
        f"They boarded at {embarked}, paid {fare_str} for their fare, "
        f"and traveled with {family_str} aboard. "
        f"They were assigned {cabin_str}."
    )


# ── 3. Write narratives to a temp CSV for KSelect.from_csv ───────────────────
def build_narrative_csv(df) -> str:
    tmp = tempfile.mkdtemp(prefix="kselect_titanic_")
    csv_path = os.path.join(tmp, "titanic_narratives.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["narrative"] + _METADATA_COLS)
        writer.writeheader()
        for row in df.to_dict("records"):
            record = {"narrative": row_to_narrative(row)}
            for col in _METADATA_COLS:
                record[col] = str(row.get(col, ""))
            writer.writerow(record)

    print(f"[prep] Wrote {len(df)} narrative rows to {csv_path}")
    return csv_path


# ── 4. Questions ──────────────────────────────────────────────────────────────
QUESTIONS = [
    "Who were the notable first class survivors and what do we know about them?",
    "Compare the survival rates of men versus women based on the passenger records.",
    "Which third class passengers survived and what were their characteristics?",
    "Tell me about passengers who boarded at Queenstown — did they fare better or worse?",
    "Which passengers paid the highest fares and did that correlate with survival?",
    "Were large families more or less likely to survive than solo travelers?",
    "Describe what happened to children under 10 on the Titanic.",
]


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        return

    df = load_titanic_df()
    csv_path = build_narrative_csv(df)

    print("\n" + "=" * 70)
    print("KSelect Titanic Stress Test")
    print("=" * 70)
    print(f"Passengers: {len(df)} | model: {MODEL} | k=15 | 3000 token context")
    print("Building FAISS + BM25 index...\n")

    from kselect import KSelect
    from kselect.llm.anthropic_client import AnthropicClient
    from kselect.models.config import KSelectConfig, LLMConfig

    cfg = KSelectConfig()
    cfg.llm = LLMConfig(max_tokens=600)

    ks = KSelect.from_csv(
        csv_path,
        text_col="narrative",
        metadata=_METADATA_COLS,
        config=cfg,
        bm25=True,
        chunk_size=150,
        chunk_overlap=0,
    )
    llm = AnthropicClient(api_key=api_key, model=MODEL)
    ks._llm = llm
    ks._assembler._llm = llm

    print(f"Index built: {ks.index_size()} chunks\n")
    print("=" * 70)

    timed_query = timer(ks.query)
    timings: list[float] = []

    for q in QUESTIONS:
        print(f"\nQuestion: {q!r}")
        t0 = time.perf_counter()
        result = timed_query(q, k=15, fast=True, max_context_tokens=3000)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        print(f"Answer:\n{result.answer}")
        print(f"\nConfidence: {result.confidence:.2f} | "
              f"Chunks in context: {result.chunks_in_context} | "
              f"Dropped: {result.chunks_dropped}")
        print(f"Retrieval: {result.retrieval_ms:.1f}ms | "
              f"LLM: {result.llm_ms/1000:.2f}s | "
              f"Total: {elapsed:.2f}s")
        print("-" * 70)

    print(f"\n{'='*70}")
    print(f"Query timing summary ({len(timings)} queries)")
    print(f"  avg: {sum(timings)/len(timings):.2f}s  "
          f"min: {min(timings):.2f}s  "
          f"max: {max(timings):.2f}s  "
          f"total: {sum(timings):.2f}s")
    print("="*70)

    save_path = "/tmp/kselect_titanic_state"
    ks.save(save_path)
    print(f"\n[saved] Index persisted to {save_path}")


if __name__ == "__main__":
    main()
