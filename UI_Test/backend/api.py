"""
KSelect UI Test — FastAPI backend.

Endpoints:
  POST /upload      — accept a CSV file + config, build a KSelect index
  POST /chat        — ask a question against the current index
  GET  /status      — index health / diagnostics
  DELETE /reset     — tear down the current index
  POST /eval/run    — stream SSE eval results (KSelect vs LangChain)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncGenerator

_executor = ThreadPoolExecutor(max_workers=4)

# Make sure the project root is on PYTHONPATH when running from UI_Test/backend/
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR = str(Path(__file__).resolve().parents[1] / "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from dotenv import load_dotenv
load_dotenv(Path(PROJECT_ROOT) / ".env")

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="KSelect UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (single-session, single-index) ────────────────────────────────

_state: dict[str, Any] = {
    "ks": None,
    "filename": None,
    "columns": [],
    "text_col": "_text (all columns)",
    "chunk_count": 0,
    "index_ms": 0.0,
}


# ── Models ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    k: int = 15
    fast: bool = True
    max_context_tokens: int = 6000


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    chunks_retrieved: int
    chunks_in_context: int
    chunks_dropped: int
    context_tokens: int
    sources: list[dict]


class StatusResponse(BaseModel):
    ready: bool
    filename: str | None
    text_col: str | None
    columns: list[str]
    chunk_count: int
    index_ms: float
    index_drift: float
    recall_estimate: float


# ── Upload ──────────────────────────────────────────────────────────────────────

def _row_to_text(row: dict) -> str:
    """
    Serialize every column as "Key: value | Key: value | ...".

    Critical design choice: use " | " not ". " as separator.
    Periods trigger NLTK sentence tokenization, which would split
    "Fare: 7.25. Survived: 0." into ["Fare: 7.25.", "Survived: 0."] —
    destroying the row's semantic unit. Pipe separators have no special
    meaning to any chunker, so the whole row stays intact.
    """
    parts = []
    for key, val in row.items():
        if val is None or str(val).strip() in ("", "nan", "NaN"):
            continue
        parts.append(f"{key}: {val}")
    return " | ".join(parts)


def _build_enriched_csv(original_csv: str, columns: list[str]) -> str:
    """
    Write a plain-text file where each row is a pipe-joined field string,
    separated by double newlines. KSelect's paragraph chunker splits on \\n\\n,
    so each CSV row becomes exactly one chunk — all fields stay together.

    Also writes the original CSV variant for LangChain baseline.
    Returns path to the enriched plain-text file (used as KSelect input).
    """
    import csv as _csv

    # Plain-text file for KSelect (paragraph chunking on \n\n)
    txt_path = original_csv.replace(".csv", "_enriched.txt")
    # CSV variant for LangChain CSVLoader
    enriched_csv_path = original_csv.replace(".csv", "_enriched.csv")

    rows_text = []
    with open(original_csv, newline="", encoding="utf-8-sig") as src:
        reader = _csv.DictReader(src)
        with open(enriched_csv_path, "w", newline="", encoding="utf-8") as dst_csv:
            writer = _csv.DictWriter(dst_csv, fieldnames=["_text"] + columns)
            writer.writeheader()
            for row in reader:
                text = _row_to_text(row)
                rows_text.append(text)
                writer.writerow({"_text": text, **row})

    Path(txt_path).write_text("\n\n".join(rows_text), encoding="utf-8")
    return enriched_csv_path   # still return CSV path for LangChain; KSelect uses txt


@app.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    chunk_size: int = Form(200),
    chunk_overlap: int = Form(20),
    bm25: bool = Form(True),
):
    """
    Upload a CSV. Every column is combined into a rich text chunk so the entire
    row is searchable — no need to pick a single text column.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set on the server.")

    # Save upload to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="kselect_ui_")
    csv_path = os.path.join(tmp_dir, file.filename)
    content = await file.read()
    Path(csv_path).write_bytes(content)

    # Detect columns
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = _csv.DictReader(fh)
        columns = list(reader.fieldnames or [])

    if not columns:
        raise HTTPException(400, "CSV has no columns.")

    # Build enriched files:
    #   _enriched.csv  — for LangChain CSVLoader baseline
    #   _enriched.txt  — for KSelect: one row per paragraph (\n\n separated)
    #                    so paragraph chunking keeps every field in one chunk
    enriched_csv_path = _build_enriched_csv(csv_path, columns)
    enriched_txt_path = csv_path.replace(".csv", "_enriched.txt")

    # Build KSelect index
    from kselect import KSelect
    from kselect.llm.anthropic_client import AnthropicClient
    from kselect.models.config import KSelectConfig, LLMConfig, ChunkingConfig

    from kselect.models.config import FusionConfig, RankingConfig, ContextConfig, ContextStrategy

    cfg = KSelectConfig()
    cfg.llm = LLMConfig(max_tokens=800)
    # Paragraph chunking: splits on \n\n — each row in the .txt is one paragraph
    cfg.chunking = ChunkingConfig(strategy="paragraph", min_chunk_length=20)
    cfg.bm25.enabled = bm25
    # BGE-large: 335M params, ~15 points higher on MTEB than all-MiniLM-L6-v2 (22M).
    # Better semantic matching for tabular field values ("Fare: 512.33", "Embarked: C").
    cfg.embedding.model = "BAAI/bge-large-en-v1.5"
    # Lower rrf_k (20 vs default 60) amplifies BM25's contribution in the fusion score.
    # For CSV data with exact keywords (names, "Cherbourg", fare values), BM25 is the
    # strongest retrieval signal — exact-match rows should rank far above irrelevant ones.
    cfg.fusion = FusionConfig(mode="rrf", rrf_k=20)
    # mmr_lambda=0.7: more diversity weight than default 0.5 — tabular rows are structurally
    # near-identical so MMR needs to work harder to surface rows with DIFFERENT field values.
    cfg.ranking = RankingConfig(mode="hybrid", mmr_lambda=0.7)
    # LOST_IN_MIDDLE + larger context: best chunks at positions 1 and last (peak LLM attention).
    cfg.context = ContextConfig(strategy=ContextStrategy.LOST_IN_MIDDLE, max_context_tokens=8000)

    def _build():
        from kselect.ingestion.loaders import FolderLoader
        from kselect.ingestion.pipeline import IngestionPipeline
        from kselect.index.faiss_index import FAISSIndex
        from kselect.index.bm25_index import BM25Index
        from kselect.index.manager import IndexManager
        from kselect.backends.local import LocalBackend
        import tempfile as _tmp

        state_dir = _tmp.mkdtemp(prefix="kselect_state_")
        mgr = IndexManager(FAISSIndex(), BM25Index(), LocalBackend(state_dir), cfg)

        loader = FolderLoader(str(Path(enriched_txt_path).parent))
        pipeline = IngestionPipeline()
        # FolderLoader only picks up supported extensions — .txt is in the list
        chunks = pipeline.run(loader, cfg)
        mgr.build(chunks)
        return KSelect(cfg, mgr)

    t0 = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        ks = await loop.run_in_executor(_executor, _build)
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Index build failed: {exc}")

    index_ms = (time.perf_counter() - t0) * 1000

    # Sonnet 4.6: significantly more capable than Haiku for grounded factual answers.
    # Faithfulness improves because Sonnet follows the "cite specific values" instruction
    # more reliably than Haiku. Judge still uses Haiku (cheap, fast, binary yes/no).
    llm = AnthropicClient(api_key=api_key, model="claude-sonnet-4-6")
    ks._llm = llm
    ks._assembler._llm = llm

    # Tear down previous temp dir
    old_tmp = _state.get("_tmp_dir")
    if old_tmp and Path(old_tmp).exists():
        shutil.rmtree(old_tmp, ignore_errors=True)

    _state.update(
        ks=ks,
        filename=file.filename,
        columns=columns,
        text_col="_text (all columns)",
        chunk_count=ks.index_size(),
        index_ms=index_ms,
        _tmp_dir=tmp_dir,
    )

    return {
        "ok": True,
        "filename": file.filename,
        "columns": columns,
        "text_col": "_text (all columns)",
        "chunk_count": ks.index_size(),
        "index_ms": round(index_ms, 1),
    }


# ── Chat ────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    ks = _state.get("ks")
    if ks is None:
        raise HTTPException(400, "No index loaded. Upload a CSV first.")

    def _query():
        return ks.query(
            req.question,
            k=req.k,
            hybrid=True,
            max_context_tokens=req.max_context_tokens,
        )

    t0 = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _query)
    except Exception as exc:
        raise HTTPException(500, f"Query failed: {exc}")

    total_ms = (time.perf_counter() - t0) * 1000

    sources = [
        {
            "chunk_id": s.chunk_id,
            "doc_id": s.doc_id,
            "snippet": s.snippet[:300],
            "score": round(s.score, 4),
            "metadata": s.metadata,
        }
        for s in result.sources
    ]

    return ChatResponse(
        answer=result.answer,
        confidence=result.confidence,
        retrieval_ms=round(result.retrieval_ms, 1),
        llm_ms=round(result.llm_ms, 1),
        total_ms=round(total_ms, 1),
        chunks_retrieved=result.chunks_retrieved,
        chunks_in_context=result.chunks_in_context,
        chunks_dropped=result.chunks_dropped,
        context_tokens=result.context_tokens,
        sources=sources,
    )


# ── Status ──────────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
async def status():
    ks = _state.get("ks")
    return StatusResponse(
        ready=ks is not None,
        filename=_state.get("filename"),
        text_col=_state.get("text_col"),
        columns=_state.get("columns", []),
        chunk_count=_state.get("chunk_count", 0),
        index_ms=round(_state.get("index_ms", 0.0), 1),
        index_drift=round(ks.index_drift(), 4) if ks else 0.0,
        recall_estimate=round(ks.recall_estimate(), 4) if ks else 0.0,
    )


# ── Reset ───────────────────────────────────────────────────────────────────────

@app.delete("/reset")
async def reset():
    old_tmp = _state.get("_tmp_dir")
    if old_tmp and Path(old_tmp).exists():
        shutil.rmtree(old_tmp, ignore_errors=True)
    _state.update(ks=None, filename=None, columns=[], text_col=None,
                  chunk_count=0, index_ms=0.0, _tmp_dir=None)
    return {"ok": True}


# ── Eval ────────────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    questions: list[str]
    k: int = 15


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_eval(questions: list[str], k: int) -> AsyncGenerator[str, None]:
    ks = _state.get("ks")
    if ks is None:
        yield _sse("error", {"message": "No index loaded. Upload a CSV first."})
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield _sse("error", {"message": "ANTHROPIC_API_KEY not set."})
        return

    # Resolve enriched CSV path for LangChain baseline
    tmp_dir = _state.get("_tmp_dir")
    filename = _state.get("filename", "upload.csv")
    enriched_csv = None
    if tmp_dir:
        candidate = os.path.join(tmp_dir, filename.replace(".csv", "_enriched.csv"))
        if Path(candidate).exists():
            enriched_csv = candidate

    import anthropic as _anthropic
    # Sync client: judge calls run inside ThreadPoolExecutor threads.
    # AsyncAnthropic + asyncio.run() caused "Event loop is closed" errors
    # that silently killed the LangChain eval phase.
    judge_client = _anthropic.Anthropic(api_key=api_key)
    JUDGE_MODEL = "claude-haiku-4-5-20251001"

    from metrics import evaluate_kselect, evaluate_langchain, MetricResult

    loop = asyncio.get_event_loop()

    # ── KSelect evaluation ─────────────────────────────────────────────────────
    yield _sse("phase", {"phase": "kselect", "total": len(questions)})

    ks_results: list[MetricResult] = []
    for i, q in enumerate(questions):
        yield _sse("progress", {"system": "kselect", "index": i, "question": q})
        def _run_ks(q=q):
            from metrics import evaluate_kselect
            import anthropic as _a
            jc = _a.Anthropic(api_key=api_key)
            return evaluate_kselect(ks, [q], jc, JUDGE_MODEL, k=k)[0]
        r = await loop.run_in_executor(_executor, _run_ks)
        ks_results.append(r)
        yield _sse("result", {"system": "kselect", "index": i, "question": q, **r.to_dict()})

    # ── LangChain evaluation ───────────────────────────────────────────────────
    if enriched_csv:
        yield _sse("phase", {"phase": "langchain", "total": len(questions)})

        lc_results: list[MetricResult] = []
        for i, q in enumerate(questions):
            yield _sse("progress", {"system": "langchain", "index": i, "question": q})
            def _run_lc(q=q):
                from metrics import evaluate_langchain
                import anthropic as _a
                jc = _a.Anthropic(api_key=api_key)
                return evaluate_langchain(enriched_csv, [q], api_key, jc, JUDGE_MODEL, k=k)[0]
            r = await loop.run_in_executor(_executor, _run_lc)
            lc_results.append(r)
            yield _sse("result", {"system": "langchain", "index": i, "question": q, **r.to_dict()})
    else:
        yield _sse("warning", {"message": "Enriched CSV not found — skipping LangChain baseline."})
        lc_results = []

    # ── Summary ────────────────────────────────────────────────────────────────
    def _avg(results: list[MetricResult], attr: str) -> float:
        vals = [getattr(r, attr) for r in results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _summary(results: list[MetricResult]) -> dict:
        if not results:
            return {}
        return {
            "faithfulness":      _avg(results, "faithfulness"),
            "answer_relevancy":  _avg(results, "answer_relevancy"),
            "context_recall":    _avg(results, "context_recall"),
            "context_precision": _avg(results, "context_precision"),
            "latency_ms":        _avg(results, "latency_ms"),
            "tokens_used":       int(_avg(results, "tokens_used")),
            "overall":           round(sum(r.overall() for r in results) / len(results), 4),
        }

    yield _sse("summary", {
        "kselect":   _summary(ks_results),
        "langchain": _summary(lc_results) if lc_results else None,
    })
    yield _sse("done", {})


@app.post("/eval/run")
async def eval_run(req: EvalRequest):
    if not req.questions:
        raise HTTPException(400, "Provide at least one question.")
    return StreamingResponse(
        _stream_eval(req.questions, req.k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
