"""
rag.py
------
Retrieval-Augmented Generation (RAG) layer for the SVA pipeline.

Two completely separate FAISS indices are maintained:
  - RTL index   : Verilog/SystemVerilog source files, chunked per module.
  - Doc index   : Design documentation (markdown, plain text), chunked by
                  paragraph with a sliding overlap window.

Keeping the indices separate lets the agent direct queries to the right
corpus (e.g. "search RTL for key-expansion logic" vs "search docs for
AES specification requirements") without contaminating results.
"""

import os
import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton encoder — load the SentenceTransformer model once and share it
# across all retriever instances to avoid duplicate GPU memory usage.
# ---------------------------------------------------------------------------
_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}


def _get_encoder(model_name: str) -> SentenceTransformer:
    """Return a shared SentenceTransformer instance (loaded once)."""
    if model_name not in _ENCODER_CACHE:
        logger.info("Loading embedding model: %s", model_name)
        _ENCODER_CACHE[model_name] = SentenceTransformer(
            model_name, trust_remote_code=True,
        )
    else:
        logger.info("Reusing cached embedding model: %s", model_name)
    return _ENCODER_CACHE[model_name]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# A chunk is a dict with two keys:
#   "text"     : the raw text content to embed and retrieve
#   "metadata" : provenance info (source file, module name, chunk index, …)
Chunk = Dict[str, Any]


# ---------------------------------------------------------------------------
# RTL chunker — module-aware splitting
# ---------------------------------------------------------------------------

# Matches a complete Verilog/SystemVerilog module from the "module" keyword
# through to "endmodule".  DOTALL lets "." match newlines.
_MODULE_RE = re.compile(r"(module\s+\w+[\s\S]*?endmodule)", re.MULTILINE)


def _build_module_prefix(module_name: str, design_info: Any = None) -> str:
    """
    Build a hierarchy-aware prefix for an RTL chunk.

    Prepends module name, parent, ports, and children so the embedding
    model has structural context alongside the code.
    """
    if design_info is None:
        return ""

    # Find parent module.
    parent = None
    hierarchy_tree = getattr(design_info, "hierarchy_tree", {})
    for p, children in hierarchy_tree.items():
        if module_name in children:
            parent = p
            break

    # Get port summary.
    modules = getattr(design_info, "modules", {})
    mod = modules.get(module_name)
    port_summary = ""
    if mod and hasattr(mod, "ports") and mod.ports:
        port_parts = []
        for p in sorted(mod.ports.values(), key=lambda x: x.name)[:10]:
            port_parts.append(f"{p.name}({p.direction[:3]},{p.width})")
        port_summary = ", ".join(port_parts)

    # Get children.
    children_list = hierarchy_tree.get(module_name, [])

    parts = [f"Module: {module_name}"]
    if parent:
        parts.append(f"Parent: {parent}")
    if port_summary:
        parts.append(f"Ports: {port_summary}")
    if children_list:
        parts.append(f"Children: {', '.join(children_list)}")

    return "[" + " | ".join(parts) + "]\n"


def build_module_summary_chunks(design_info: Any) -> List[Chunk]:
    """
    Create one compact summary chunk per module from DesignInfo.

    These feed into the Stage 1 (coarse) index of hierarchical retrieval.
    Each chunk contains the module name, parent, ports, and submodules
    — enough for the retriever to identify which module is relevant to a query.
    """
    chunks: List[Chunk] = []
    modules = getattr(design_info, "modules", {})
    hierarchy_tree = getattr(design_info, "hierarchy_tree", {})

    for mod_name, mod_info in modules.items():
        # Find parent.
        parent = None
        for p, children in hierarchy_tree.items():
            if mod_name in children:
                parent = p
                break

        lines = [f"Module: {mod_name}"]
        if parent:
            lines.append(f"Parent: {parent}")

        if hasattr(mod_info, "ports") and mod_info.ports:
            port_strs = []
            for p in sorted(mod_info.ports.values(), key=lambda x: x.name):
                port_strs.append(f"{p.name}({p.direction},{p.width})")
            lines.append(f"Ports: {', '.join(port_strs)}")

        children_list = hierarchy_tree.get(mod_name, [])
        if children_list:
            lines.append(f"Submodules: {', '.join(children_list)}")
        else:
            lines.append("Submodules: none (leaf module)")

        if hasattr(mod_info, "cells") and mod_info.cells:
            inst_strs = [
                f"{name}->{typ}"
                for name, typ in sorted(mod_info.cells.items())[:20]
            ]
            lines.append(f"Instances: {', '.join(inst_strs)}")

        chunks.append({
            "text": "\n".join(lines),
            "metadata": {
                "module": mod_name,
                "type": "module_summary",
            },
        })

    logger.info("Built %d module summary chunk(s).", len(chunks))
    return chunks


def chunk_rtl_file(file_path: str, max_chars: int = 8000, design_info: Any = None) -> List[Chunk]:
    """
    Split a single Verilog/SystemVerilog file into module-level chunks.

    Each chunk covers exactly one `module … endmodule` block, preserving
    port declarations, parameter lists, and internal logic together.  This
    is critical because SVA assertions reference exact signal names and
    widths — a chunk that splits a port declaration from its always block
    would mislead the retriever.

    If a module body exceeds `max_chars` (e.g. a large parameterised module)
    it is further split into overlapping sub-chunks of that size so that
    embeddings remain meaningful and context windows are not overflowed.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the .v / .sv file.
    max_chars : int
        Maximum character length per chunk before secondary splitting.

    Returns
    -------
    List[Chunk]
        One or more chunks, each with 'text' and 'metadata' keys.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    chunks: List[Chunk] = []
    matches = list(_MODULE_RE.finditer(content))

    if not matches:
        # File has no module keyword (e.g. a header / package file).
        # Treat the whole file as a single chunk.
        chunks.append({
            "text": content.strip(),
            "metadata": {
                "source": file_path,
                "module": Path(file_path).stem,
                "type": "rtl",
                "chunk_index": 0,
            },
        })
        return chunks

    for match in matches:
        module_text = match.group(1)

        # Extract the module name from the first line for metadata.
        name_match = re.match(r"module\s+(\w+)", module_text)
        module_name = name_match.group(1) if name_match else Path(file_path).stem

        # Prepend hierarchy metadata for the embedding model.
        prefix = _build_module_prefix(module_name, design_info)

        if len(module_text) <= max_chars:
            # Module fits in one chunk — ideal case.
            chunks.append({
                "text": prefix + module_text,
                "metadata": {
                    "source": file_path,
                    "module": module_name,
                    "type": "rtl",
                    "chunk_index": 0,
                },
            })
        else:
            # Module is too large; split into overlapping sub-chunks.
            # Use a 200-char overlap so logic near boundaries isn't lost.
            overlap = 200
            start = 0
            sub_idx = 0
            while start < len(module_text):
                end = min(start + max_chars, len(module_text))
                chunks.append({
                    "text": prefix + module_text[start:end],
                    "metadata": {
                        "source": file_path,
                        "module": module_name,
                        "type": "rtl",
                        "chunk_index": sub_idx,
                    },
                })
                start = end - overlap if end < len(module_text) else end
                sub_idx += 1

    return chunks


def load_rtl_chunks(rtl_dir: str, max_chars: int = 8000, design_info: Any = None) -> List[Chunk]:
    """
    Walk `rtl_dir` recursively and chunk every .v and .sv file found.

    Returns
    -------
    List[Chunk]
        All RTL chunks across all files.
    """
    all_chunks: List[Chunk] = []
    rtl_extensions = {".v", ".sv"}

    for root, _, files in os.walk(rtl_dir):
        for fname in sorted(files):  # sorted for reproducible ordering
            if Path(fname).suffix in rtl_extensions:
                fpath = os.path.join(root, fname)
                file_chunks = chunk_rtl_file(fpath, max_chars=max_chars, design_info=design_info)
                all_chunks.extend(file_chunks)
                logger.info(
                    "RTL: %s → %d chunk(s)", fname, len(file_chunks)
                )

    logger.info("Total RTL chunks: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Documentation chunker — paragraph-aware splitting
# ---------------------------------------------------------------------------

def chunk_document_file(
    file_path: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Split a documentation file into overlapping text chunks.

    The splitter prefers to break at paragraph boundaries (double newline)
    so that semantically related sentences stay together.  If no paragraph
    break is found within the window it falls back to a single newline,
    then to a hard character cut.

    Parameters
    ----------
    file_path : str
        Path to a .md or .txt documentation file.
    chunk_size : int
        Target maximum character length per chunk.
    overlap : int
        Number of characters to repeat at the start of the next chunk
        to preserve cross-boundary context.

    Returns
    -------
    List[Chunk]
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read().strip()
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    if not content:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_idx = 0
    iteration = 0

    while start < len(content):
        iteration += 1
        if iteration > 1000:
            logger.error(
                "chunk_document_file stuck: file=%s len=%d start=%d chunks=%d",
                file_path, len(content), start, len(chunks),
            )
            break
        end = min(start + chunk_size, len(content))

        # Try to snap the boundary to a paragraph break within the window.
        if end < len(content):
            # Require the boundary to advance by at least `overlap` past
            # start; otherwise start doesn't move forward on the next
            # iteration (`start = end - overlap`) and we loop forever on
            # files with frequent paragraph breaks.
            min_end = start + overlap + 1
            para_break = content.rfind("\n\n", start, end)
            if para_break != -1 and para_break > min_end:
                end = para_break
            else:
                # Fall back to a single newline.
                line_break = content.rfind("\n", start, end)
                if line_break != -1 and line_break > min_end:
                    end = line_break

        chunk_text = content[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": file_path,
                    "type": "documentation",
                    "chunk_index": chunk_idx,
                    "char_start": start,
                },
            })
            chunk_idx += 1

        # Slide the window forward, keeping `overlap` chars of context.
        start = end - overlap if end < len(content) else end

    return chunks


def load_doc_chunks(
    docs_dir: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Walk `docs_dir` and chunk every .md and .txt file.

    Returns
    -------
    List[Chunk]
    """
    all_chunks: List[Chunk] = []
    doc_extensions = {".md", ".txt", ".pdf"}  # .pdf support can be added later

    if not os.path.isdir(docs_dir):
        logger.warning("Docs directory not found: %s", docs_dir)
        return []

    for root, _, files in os.walk(docs_dir):
        for fname in sorted(files):
            if Path(fname).suffix in doc_extensions:
                fpath = os.path.join(root, fname)
                file_chunks = chunk_document_file(
                    fpath, chunk_size=chunk_size, overlap=overlap
                )
                all_chunks.extend(file_chunks)
                logger.info(
                    "Doc: %s → %d chunk(s)", fname, len(file_chunks)
                )

    logger.info("Total doc chunks: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# FAISS retriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Thin wrapper around a FAISS flat inner-product index.

    Workflow:
      1. Call `build(chunks)` to embed all chunks and build the index.
      2. Call `save(path_prefix)` to persist the index + metadata to disk.
      3. On subsequent runs, call `load(path_prefix)` to skip re-embedding.
      4. Call `retrieve(query, k)` to get the top-k most similar chunks.

    The embeddings are L2-normalised before insertion so that inner-product
    search is equivalent to cosine similarity — appropriate for sentence
    embeddings whose directions carry the semantic signal.
    """

    def __init__(self, embedding_model_name: str):
        """
        Parameters
        ----------
        embedding_model_name : str
            HuggingFace model name passed to SentenceTransformer.
            e.g. "sentence-transformers/all-MiniLM-L6-v2"
        """
        # Use the shared singleton encoder to avoid loading the model
        # multiple times into GPU memory.
        self.encoder = _get_encoder(embedding_model_name)
        self.index: Optional[faiss.Index] = None
        # Parallel list to the FAISS index — stores text + metadata for
        # each vector so we can return human-readable results.
        self.chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """
        Embed `chunks` and populate the FAISS index.

        Parameters
        ----------
        chunks : List[Chunk]
            Each chunk must have a "text" key containing the string to embed.
        batch_size : int
            Number of texts to encode in each GPU/CPU batch.
        """
        if not chunks:
            logger.warning("build() called with empty chunk list — index will be empty.")
            self.chunks = []
            return

        texts = [c["text"] for c in chunks]
        logger.info("Encoding %d chunks (batch_size=%d) …", len(texts), batch_size)

        # encode() returns a numpy float32 array of shape (N, dim).
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        # L2-normalise so cosine similarity = inner product.
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        # IndexFlatIP performs exact brute-force inner-product search.
        # For typical design corpora (hundreds to low thousands of chunks)
        # this is fast enough; switch to IndexIVFFlat for very large corpora.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, dim)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path_prefix: str) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Two files are written:
          {path_prefix}.faiss  — the binary FAISS index
          {path_prefix}.pkl    — pickled list of Chunk dicts
        """
        if self.index is None:
            raise RuntimeError("Cannot save — index has not been built yet.")

        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path_prefix + ".faiss")
        with open(path_prefix + ".pkl", "wb") as fh:
            pickle.dump(self.chunks, fh)
        logger.info("Index saved to %s.faiss / .pkl", path_prefix)

    def load(self, path_prefix: str) -> None:
        """
        Load a previously persisted index from disk.

        Parameters
        ----------
        path_prefix : str
            Same prefix string that was used in `save()`.
        """
        faiss_path = path_prefix + ".faiss"
        pkl_path = path_prefix + ".pkl"

        if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Index files not found at {path_prefix}. "
                "Run with force_rebuild_index=True to build them."
            )

        self.index = faiss.read_index(faiss_path)
        with open(pkl_path, "rb") as fh:
            self.chunks = pickle.load(fh)
        logger.info(
            "Index loaded from %s: %d vectors", path_prefix, self.index.ntotal
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return the top-k chunks most similar to `query`.

        Parameters
        ----------
        query : str
            Natural-language or code-snippet query.
        k : int
            Number of results to return.

        Returns
        -------
        List[dict]
            Each result has keys: "text", "metadata", "score".
            Results are ordered by descending cosine similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("retrieve() called on empty index — returning []")
            return []

        # Embed the query using the same model and normalise.
        query_emb = self.encoder.encode(
            [query], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_emb)

        # Search returns arrays of shape (1, k).
        scores, indices = self.index.search(query_emb, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 when fewer than k results exist
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(score),
            })

        return results


# ---------------------------------------------------------------------------
# Convenience builder used by main.py
# ---------------------------------------------------------------------------

def build_or_load_retriever(
    index_path: str,
    chunks: List[Chunk],
    embedding_model: str,
    force_rebuild: bool = False,
    batch_size: int = 64,
) -> FAISSRetriever:
    """
    Return a ready-to-use FAISSRetriever.

    If `index_path` files already exist and `force_rebuild` is False, the
    index is loaded from disk (fast path).  Otherwise the index is built
    from `chunks` and saved for next time.

    Parameters
    ----------
    index_path : str
        Path prefix for the .faiss / .pkl files.
    chunks : List[Chunk]
        Source chunks — only used when building (not loading).
    embedding_model : str
        SentenceTransformer model name.
    force_rebuild : bool
        If True, rebuild even when cached files exist.
    """
    retriever = FAISSRetriever(embedding_model)

    faiss_exists = (
        os.path.exists(index_path + ".faiss")
        and os.path.exists(index_path + ".pkl")
    )

    if faiss_exists and not force_rebuild:
        logger.info("Loading cached index from %s", index_path)
        retriever.load(index_path)
    else:
        logger.info("Building index at %s …", index_path)
        retriever.build(chunks, batch_size=batch_size)
        retriever.save(index_path)

    return retriever


# ---------------------------------------------------------------------------
# Hybrid BM25 + Dense retriever (Improvement 3)
# ---------------------------------------------------------------------------

def _tokenize_for_bm25(text: str) -> List[str]:
    """
    Simple tokenizer for BM25 that handles Verilog naming conventions.

    Splits on whitespace, underscores, and camelCase boundaries, then
    lowercases everything.  This ensures that a query for "encrypt" matches
    a chunk containing "AES_Encrypt" or "aes_encrypt_out".
    """
    import re as _re
    # Split camelCase: insert space before uppercase letters preceded by lowercase.
    text = _re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on non-alphanumeric characters (underscores, punctuation, whitespace).
    tokens = _re.split(r"[^a-zA-Z0-9]+", text)
    return [t.lower() for t in tokens if t]


class HybridRetriever:
    """
    Combines FAISS dense retrieval with BM25 keyword retrieval using
    Reciprocal Rank Fusion (RRF) to merge results.

    Exposes the same interface as :class:`FAISSRetriever` so it can be used
    as a drop-in replacement anywhere a retriever is expected.

    Parameters
    ----------
    embedding_model_name : str
        HuggingFace model name for the dense encoder.
    rrf_k : int
        RRF smoothing constant.  Higher values reduce the advantage of
        top-ranked results.  Standard default is 60.
    """

    def __init__(self, embedding_model_name: str, rrf_k: int = 60):
        from rank_bm25 import BM25Okapi

        self.rrf_k = rrf_k

        # Dense component — delegates to FAISSRetriever.
        self._dense = FAISSRetriever(embedding_model_name)

        # BM25 component — built lazily in build() or load().
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_class = BM25Okapi

        # Shared chunk list — identical to the one in _dense.
        self.chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """
        Build both the FAISS dense index and the BM25 keyword index over
        the same chunks.
        """
        # Build the dense index.
        self._dense.build(chunks, batch_size=batch_size)
        self.chunks = chunks

        # Build the BM25 index from tokenised chunk texts.
        tokenised_corpus = [_tokenize_for_bm25(c["text"]) for c in chunks]
        self._bm25 = self._bm25_class(tokenised_corpus)

        logger.info(
            "Hybrid index built: %d chunks (FAISS + BM25)", len(chunks)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path_prefix: str) -> None:
        """
        Persist both indices to disk.

        Files written:
          {path_prefix}.faiss  — FAISS binary index
          {path_prefix}.pkl    — pickled chunks + BM25 state
        """
        self._dense.save(path_prefix)
        # Overwrite the .pkl file to include BM25 state alongside chunks.
        import pickle
        with open(path_prefix + ".pkl", "wb") as fh:
            pickle.dump({"chunks": self.chunks, "bm25": self._bm25}, fh)
        logger.info("Hybrid index saved to %s.faiss / .pkl", path_prefix)

    def load(self, path_prefix: str) -> None:
        """
        Load both indices from disk.
        """
        import pickle

        faiss_path = path_prefix + ".faiss"
        pkl_path = path_prefix + ".pkl"

        if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Hybrid index files not found at {path_prefix}."
            )

        # Load FAISS index.
        self._dense.index = faiss.read_index(faiss_path)

        # Load chunks + BM25 state.
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        # Handle both old-format (plain list) and new-format (dict with bm25).
        if isinstance(data, dict) and "bm25" in data:
            self.chunks = data["chunks"]
            self._dense.chunks = data["chunks"]
            self._bm25 = data["bm25"]
        else:
            # Old format — just a chunk list; rebuild BM25 on the fly.
            self.chunks = data if isinstance(data, list) else []
            self._dense.chunks = self.chunks
            tokenised = [_tokenize_for_bm25(c["text"]) for c in self.chunks]
            self._bm25 = self._bm25_class(tokenised)
            logger.info("BM25 index rebuilt from old-format .pkl")

        logger.info(
            "Hybrid index loaded from %s: %d vectors",
            path_prefix, self._dense.index.ntotal,
        )

    # ------------------------------------------------------------------
    # Retrieval with Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return the top-k chunks by fusing FAISS and BM25 results via RRF.

        1. Retrieve 2*k candidates from each retriever.
        2. Assign RRF scores: score(doc) = sum(1 / (rrf_k + rank))
        3. Return the top k by fused score.
        """
        if not self.chunks:
            return []

        fetch_k = min(2 * k, len(self.chunks))

        # --- Dense retrieval ---
        dense_results = self._dense.retrieve(query, k=fetch_k)
        # Map chunk text -> dense rank (0-based).
        dense_ranks: Dict[int, int] = {}
        for rank, r in enumerate(dense_results):
            # Find the chunk index by matching text (fast for small corpora).
            for idx, c in enumerate(self.chunks):
                if c["text"] == r["text"]:
                    dense_ranks[idx] = rank
                    break

        # --- BM25 retrieval ---
        bm25_ranks: Dict[int, int] = {}
        if self._bm25 is not None:
            query_tokens = _tokenize_for_bm25(query)
            bm25_scores = self._bm25.get_scores(query_tokens)
            # Get top fetch_k indices by descending BM25 score.
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:fetch_k]
            for rank, idx in enumerate(top_bm25_indices):
                if bm25_scores[idx] > 0:  # skip zero-score documents
                    bm25_ranks[idx] = rank

        # --- Reciprocal Rank Fusion ---
        all_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        fused: List[tuple] = []
        for idx in all_indices:
            score = 0.0
            if idx in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[idx])
            fused.append((idx, score))

        # Sort by descending fused score and take top k.
        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in fused[:k]:
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": score,
            })

        return results


# ---------------------------------------------------------------------------
# Convenience builder for HybridRetriever
# ---------------------------------------------------------------------------

def build_or_load_hybrid_retriever(
    index_path: str,
    chunks: List[Chunk],
    embedding_model: str,
    rrf_k: int = 60,
    force_rebuild: bool = False,
    batch_size: int = 64,
) -> HybridRetriever:
    """
    Return a ready-to-use HybridRetriever.

    Same caching logic as :func:`build_or_load_retriever`: loads from disk
    if index files exist, otherwise builds and saves.
    """
    retriever = HybridRetriever(embedding_model, rrf_k=rrf_k)

    faiss_exists = (
        os.path.exists(index_path + ".faiss")
        and os.path.exists(index_path + ".pkl")
    )

    if faiss_exists and not force_rebuild:
        logger.info("Loading cached hybrid index from %s", index_path)
        retriever.load(index_path)
    else:
        logger.info("Building hybrid index at %s …", index_path)
        retriever.build(chunks, batch_size=batch_size)
        retriever.save(index_path)

    return retriever


# ---------------------------------------------------------------------------
# Hierarchical two-stage retriever
# ---------------------------------------------------------------------------

class HierarchicalRetriever:
    """
    Two-stage retriever: module summary (coarse) → full code (fine).

    Stage 1: Query the module summary index to find the top-k1 relevant
    modules by name, ports, and hierarchy position.

    Stage 2: Query the full code index, filtering results to only chunks
    from the modules identified in Stage 1.

    Exposes the same ``retrieve(query, k)`` interface as FAISSRetriever
    and HybridRetriever — drop-in replacement.
    """

    def __init__(
        self,
        summary_retriever,
        code_retriever,
        stage1_k: int = 5,
        stage2_k: int = 5,
    ):
        self.summary_retriever = summary_retriever
        self.code_retriever = code_retriever
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.chunks = code_retriever.chunks

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Two-stage retrieval with module filtering."""
        module_results = self.summary_retriever.retrieve(query, k=self.stage1_k)
        relevant_modules = {
            r["metadata"].get("module", "")
            for r in module_results if r["metadata"].get("module")
        }

        if not relevant_modules:
            return self.code_retriever.retrieve(query, k=k)

        logger.debug("Hierarchical Stage 1: matched %s", relevant_modules)

        fetch_k = min(k * 4, len(self.chunks) if self.chunks else k * 4)
        all_results = self.code_retriever.retrieve(query, k=fetch_k)

        filtered = [
            r for r in all_results
            if r["metadata"].get("module", "") in relevant_modules
        ]

        if len(filtered) < k:
            seen = {r["text"][:100] for r in filtered}
            for r in all_results:
                if r["text"][:100] not in seen:
                    filtered.append(r)
                    if len(filtered) >= k:
                        break

        return filtered[:k]


def build_or_load_hierarchical_retriever(
    summary_index_path: str,
    code_index_path: str,
    summary_chunks: List[Chunk],
    code_chunks: List[Chunk],
    rtl_embedding_model: str,
    stage1_k: int = 5,
    stage2_k: int = 5,
    use_hybrid: bool = True,
    rrf_k: int = 60,
    force_rebuild: bool = False,
) -> HierarchicalRetriever:
    """Build or load both stages of the hierarchical retriever."""
    summary_retriever = FAISSRetriever(rtl_embedding_model)
    summary_exists = (
        os.path.exists(summary_index_path + ".faiss")
        and os.path.exists(summary_index_path + ".pkl")
    )
    if summary_exists and not force_rebuild:
        summary_retriever.load(summary_index_path)
    else:
        summary_retriever.build(summary_chunks)
        summary_retriever.save(summary_index_path)

    if use_hybrid:
        code_retriever = build_or_load_hybrid_retriever(
            index_path=code_index_path, chunks=code_chunks,
            embedding_model=rtl_embedding_model, rrf_k=rrf_k,
            force_rebuild=force_rebuild,
        )
    else:
        code_retriever = build_or_load_retriever(
            index_path=code_index_path, chunks=code_chunks,
            embedding_model=rtl_embedding_model,
            force_rebuild=force_rebuild,
        )

    logger.info(
        "Hierarchical retriever: %d summaries (Stage 1), %d chunks (Stage 2).",
        len(summary_chunks), len(code_chunks),
    )
    return HierarchicalRetriever(
        summary_retriever=summary_retriever,
        code_retriever=code_retriever,
        stage1_k=stage1_k,
        stage2_k=stage2_k,
    )
