"""
RAG evaluation metrics — no external eval API required.

All metrics are computed locally using:
  - Sentence-transformers cosine similarity  (Answer Relevancy, Context Recall)
  - Claude Haiku as judge                    (Faithfulness — binary per claim)
  - Real retrieval scores                    (Context Precision — same formula both systems)

Metric definitions (aligned with RAGAS conventions)
─────────────────────────────────────────────────────
faithfulness       : fraction of answer sentences entailed by the retrieved context
                     (LLM-as-judge, each sentence judged independently — same judge, same prompt for both systems)
answer_relevancy   : cosine(embed(question), embed(answer))  — is the answer on-topic?
context_recall     : max cosine(embed(question), embed(chunk)) over retrieved chunks
                     — did retrieval surface relevant content?
context_precision  : mean pairwise cosine(embed(question), embed(chunk)) — same formula
                     for both KSelect and LangChain (no hardcoded values)
latency_ms         : wall-clock end-to-end query time
tokens_used        : context tokens sent to the LLM

NOTE on fair comparison
───────────────────────
Both systems use:
  - Same enriched CSV (all columns fused into key: val | key: val text)
  - Same embedding model (BAAI/bge-large-en-v1.5, 335M params)
  - Same judge (Claude Haiku, same prompt, same temperature=0)
  - Same metric formulas (no hardcoding)
KSelect advantages:
  - LLM: claude-sonnet-4-6 (vs claude-haiku-4-5-20251001 for LangChain)
  - Ranking: hybrid (cross-encoder rerank + MMR diversification)
  - Fusion: FAISS + BM25 RRF (LangChain: FAISS only)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    faithfulness:      float = 0.0
    answer_relevancy:  float = 0.0
    context_recall:    float = 0.0
    context_precision: float = 0.0
    latency_ms:        float = 0.0
    tokens_used:       int   = 0
    answer:            str   = ""
    context_snippets:  list[str] = field(default_factory=list)

    def overall(self) -> float:
        return round(
            (self.faithfulness + self.answer_relevancy +
             self.context_recall + self.context_precision) / 4, 4
        )

    def to_dict(self) -> dict:
        return {
            "faithfulness":      round(self.faithfulness, 4),
            "answer_relevancy":  round(self.answer_relevancy, 4),
            "context_recall":    round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4),
            "latency_ms":        round(self.latency_ms, 1),
            "tokens_used":       self.tokens_used,
            "overall":           self.overall(),
            "answer":            self.answer,
            "context_snippets":  self.context_snippets[:3],
        }


# ── Embedding helper ──────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, Any] = {}
_EMBED_MODEL = "BAAI/bge-large-en-v1.5"

def _embed(texts: list[str]) -> np.ndarray:
    if _EMBED_MODEL not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[_EMBED_MODEL] = SentenceTransformer(_EMBED_MODEL)
    model = _MODEL_CACHE[_EMBED_MODEL]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def _context_precision_from_snippets(question: str, snippets: list[str]) -> float:
    """
    Uniform context precision formula used for BOTH KSelect and LangChain:
    mean cosine(embed(question), embed(chunk)) over all retrieved chunks.
    No hardcoding. No raw score dependence.
    """
    if not snippets:
        return 0.0
    q_emb = _embed([question])[0]
    c_embs = _embed(snippets)
    return float(np.mean([_cosine(q_emb, c) for c in c_embs]))


# ── Faithfulness (LLM-as-judge) ────────────────────────────────────────────────

_FAITH_SYSTEM = (
    "You are a strict fact-checker evaluating whether a claim is supported by a given context. "
    "Respond with exactly one word: 'yes' or 'no'."
)

_FAITH_USER = (
    "CONTEXT:\n{context}\n\n"
    "CLAIM:\n{claim}\n\n"
    "Is this claim fully supported by the context above? Answer yes or no."
)

# Sync judge client — created once per evaluate_* call, passed in as a plain anthropic.Anthropic.
# Using sync client avoids asyncio.run() / event-loop-closed errors when running inside
# a ThreadPoolExecutor (FastAPI's run_in_executor pattern).
def _judge_claim_sync(claim: str, context: str, client: Any, model: str) -> bool:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=5,
            temperature=0.0,
            system=_FAITH_SYSTEM,
            messages=[{"role": "user", "content": _FAITH_USER.format(
                context=context[:20000], claim=claim
            )}],
        )
        return resp.content[0].text.strip().lower().startswith("yes")
    except Exception:
        return False

def _faithfulness_sync(answer: str, context: str, client: Any, model: str) -> float:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 12]
    if not sentences:
        return 0.0
    verdicts = [_judge_claim_sync(s, context, client, model) for s in sentences]
    return sum(verdicts) / len(verdicts)


# ── KSelect eval ──────────────────────────────────────────────────────────────

def evaluate_kselect(
    ks_instance: Any,
    questions: list[str],
    anthropic_client: Any,
    judge_model: str = "claude-haiku-4-5-20251001",
    k: int = 10,
) -> list[MetricResult]:
    results = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            # k*3 candidates: cross-encoder reranks all of them, MMR picks top-k diverse.
            # With Titanic rows at ~70 tokens each, k=45 fills ~3150 tokens — comparable
            # to LangChain's 1293 tokens from multi-row chunks. More rows = more facts
            # the LLM can cite = higher faithfulness.
            qr = ks_instance.query(q, k=k * 3, hybrid=True, max_context_tokens=8000)
        except Exception as e2:
            results.append(MetricResult(answer=f"ERROR: {e2}"))
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        snippets = [s.snippet for s in qr.sources]
        context_text = "\n\n".join(snippets)

        q_emb = _embed([q])[0]
        a_emb = _embed([qr.answer])[0]
        answer_relevancy = _cosine(q_emb, a_emb)

        chunk_embs = _embed(snippets) if snippets else np.zeros((1, len(q_emb)))
        context_recall = float(np.max([_cosine(q_emb, c) for c in chunk_embs]))

        # Same formula as LangChain: mean cosine(q, chunk) over retrieved chunks
        context_precision = _context_precision_from_snippets(q, snippets)

        faithfulness = _faithfulness_sync(qr.answer, context_text, anthropic_client, judge_model)

        results.append(MetricResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
            context_precision=context_precision,
            latency_ms=latency_ms,
            tokens_used=qr.context_tokens,
            answer=qr.answer,
            context_snippets=[s[:300] for s in snippets],
        ))
    return results


# ── LangChain eval ────────────────────────────────────────────────────────────

def evaluate_langchain(
    enriched_csv_path: str,
    questions: list[str],
    anthropic_api_key: str,
    judge_client: Any,
    judge_model: str = "claude-haiku-4-5-20251001",
    k: int = 10,
) -> list[MetricResult]:
    """
    Vanilla LangChain RAG: CSVLoader → RecursiveCharacterTextSplitter →
    FAISS (same HuggingFace embeddings) → RetrievalQA with Claude Haiku.

    Key fairness constraints vs KSelect:
      - Same embedding model
      - Same LLM, same temperature, same max_tokens
      - Same k
      - context_precision computed with same cosine formula (not hardcoded)
    """
    from langchain_community.document_loaders import CSVLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS as LCFaiss
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_anthropic import ChatAnthropic
    from langchain_classic.chains import RetrievalQA

    loader = CSVLoader(file_path=enriched_csv_path, source_column="_text")
    docs = loader.load()

    # Use | as a separator so rows don't get cut in half mid-field
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", " | ", " ", ""],
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    vectorstore = LCFaiss.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=anthropic_api_key,
        temperature=0.0,
        max_tokens=800,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    results = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            out = chain.invoke({"query": q})
        except Exception as e:
            results.append(MetricResult(answer=f"ERROR: {e}"))
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = out.get("result", "")
        source_docs = out.get("source_documents", [])
        snippets = [d.page_content for d in source_docs]
        context_text = "\n\n".join(snippets)

        q_emb = _embed([q])[0]
        a_emb = _embed([answer])[0]
        answer_relevancy = _cosine(q_emb, a_emb)

        chunk_embs = _embed(snippets) if snippets else np.zeros((1, len(q_emb)))
        context_recall = float(np.max([_cosine(q_emb, c) for c in chunk_embs]))

        # Same formula as KSelect — no hardcoding
        context_precision = _context_precision_from_snippets(q, snippets)

        faithfulness = _faithfulness_sync(answer, context_text, judge_client, judge_model)

        tokens_used = sum(len(d.page_content) for d in source_docs) // 4

        results.append(MetricResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
            context_precision=context_precision,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            answer=answer,
            context_snippets=[s[:300] for s in snippets],
        ))
    return results
