"""
main.py
-------
Entry point for the SVA generation pipeline.

Usage
-----
    python main.py project.yaml

That's it.  The YAML file contains all configuration — RTL paths, model
settings, output locations, and optional threat model.  Everything else
(hierarchy, signal map, assertion style, task description) is auto-detected.

See docs/pipeline_architecture.md for the full architecture documentation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from sva_pipeline.config_loader import (
    load_project_config,
    auto_detect,
    generate_task,
    compute_source_checksum,
    is_index_stale,
    save_checksum,
)
from sva_pipeline.slang_frontend import build_design_info
from sva_pipeline.analysis_export import (
    resolve_analysis_dir, ensure_dir,
    export_hierarchy, export_rtl_facts_json,
    export_per_module_csv, AssertionTracer,
)

# Heavy imports (torch, transformers, sentence_transformers, faiss) are
# deferred to avoid loading ~4 GB of libraries at startup.  They are
# imported lazily at the point they're needed.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_full_rtl_context(rtl_dir: str, threshold: int) -> Optional[str]:
    """
    Read all .v/.sv files from rtl_dir and concatenate them.

    Returns the concatenated string if total chars < threshold, else None.
    """
    parts = []
    total_chars = 0

    for root, _, files in os.walk(rtl_dir):
        for fname in sorted(files):
            if Path(fname).suffix in {".v", ".sv"}:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                except OSError:
                    continue
                total_chars += len(content)
                if total_chars > threshold:
                    return None
                parts.append(f"// ===== FILE: {fname} =====\n{content}")

    return "\n\n".join(parts) if parts else None


def write_output(sva_code: str, output_path: str, log_path: str) -> None:
    """Write the generated SVA to the output .sv file and the log."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("// Auto-generated SVA assertions by sva_pipeline\n")
        fh.write("// Validated with pyslang\n\n")
        fh.write(sva_code)
        fh.write("\n")

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("=" * 80 + "\n")
        fh.write(f"OUTPUT FILE : {output_path}\n")
        fh.write(f"CHAR COUNT  : {len(sva_code)}\n")
        fh.write("=" * 80 + "\n\n")
        fh.write(sva_code)
        fh.write("\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Single argument: the YAML config file path -----------------------
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python main.py <project.yaml>")
        print()
        print("The YAML file must contain at minimum:")
        print()
        print("  design:")
        print('    rtl_dir: "./path/to/verilog"')
        print('    top_module: "ModuleName"')
        print()
        print("See docs/pipeline_architecture.md for all options.")
        sys.exit(0 if "--help" in sys.argv else 1)

    yaml_path = sys.argv[1]

    # ---- Load configuration from YAML ------------------------------------
    config = load_project_config(yaml_path)

    # ---- Setup logging from config ---------------------------------------
    # Force unbuffered output so log lines appear immediately in the log
    # file even when stdout is redirected (e.g. via nohup).
    handler = logging.StreamHandler(sys.stdout)
    handler.flush = lambda: sys.stdout.flush()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
        force=True,
    )
    logger = logging.getLogger("main")

    logger.info("Pipeline config loaded from %s", yaml_path)
    logger.info("  design     : %s (top: %s)", config.rtl_dir, config.top_module)
    if config.top_file:
        logger.info("  top file   : %s", config.top_file)
    logger.info("  model      : %s (%s)", config.model_id, config.dtype)
    logger.info("  output     : %s", config.output_sva_file)
    if config.docs_dir:
        logger.info("  docs       : %s", config.docs_dir)
    if config.threat_model:
        logger.info("  threat model: %s", config.threat_model)

    # ---- Extract design info (replaces hierarchy.txt + signal_map.json) ---
    logger.info("Analysing design with Slang …")
    design_info = build_design_info(
        config.rtl_dir, config.top_module, config.top_file,
        allow_pyslang_warnings=config.allow_pyslang_warnings,
    )

    # ---- Auto-detect remaining config values -----------------------------
    config = auto_detect(config, design_info)

    logger.info("  assertion style : %s",
                "immediate" if config.reject_assert_property else "concurrent")
    if design_info.clock_signal:
        logger.info("  clock           : %s", design_info.clock_signal)
    if design_info.reset_signal:
        logger.info("  reset           : %s", design_info.reset_signal)

    # ---- Auto-generate task if not provided in YAML ----------------------
    if not config.task:
        config.task = generate_task(
            top_module=config.top_module,
            signal_map=design_info.signal_map,
            has_clock=design_info.has_clock,
            reset_signal=design_info.reset_signal,
            module_count=len(design_info.modules),
            has_threat_model=bool(config.threat_model),
        )
        logger.info("Task auto-generated (%d chars).", len(config.task))

    # ---- HTML-to-Markdown conversion (if configured) --------------------
    if config.html_docs_enabled and config.html_docs_files:
        from sva_pipeline.html2md import convert_all_html_docs
        converted = convert_all_html_docs(config)
        if converted:
            logger.info("Converted %d HTML file(s) to markdown.", len(converted))

    # ---- Full context injection for small designs ------------------------
    full_rtl_context = None
    if config.use_full_context_injection:
        full_rtl_context = load_full_rtl_context(
            config.rtl_dir, config.context_injection_threshold,
        )
        if full_rtl_context:
            logger.info("Full RTL injected into context (%d chars).", len(full_rtl_context))

    # ---- Build / reload retrieval indices --------------------------------
    # When full RTL context is injected, we can also inject small doc files
    # directly into the system prompt — no embedding model or FAISS needed.
    # This saves ~500 MB GPU + ~1 GB CPU that the SentenceTransformer would
    # consume, leaving more headroom for the 16 GB LLM.
    rtl_retriever = None
    doc_retriever = None
    full_docs_context = None

    # Load documentation text.  We read the raw files here WITHOUT importing
    # the heavy RAG module (which pulls in torch, faiss, sentence_transformers).
    # If the corpus is small enough for context injection, we skip RAG entirely.
    has_docs = bool(config.docs_dir) or bool(config.threat_model)
    all_doc_text = ""

    if has_docs:
        doc_parts = []
        if config.docs_dir:
            for root, _, files in os.walk(config.docs_dir):
                for fname in sorted(files):
                    if Path(fname).suffix in {".md", ".txt"}:
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                                doc_parts.append(fh.read())
                            logger.info("  Doc: %s (%d chars)", fname, len(doc_parts[-1]))
                        except OSError:
                            pass

        if config.threat_model and os.path.exists(config.threat_model):
            logger.info("  Threat model: %s", config.threat_model)
            with open(config.threat_model, "r", encoding="utf-8", errors="ignore") as fh:
                doc_parts.append(fh.read())

        all_doc_text = "\n\n---\n\n".join(doc_parts)
        logger.info("Doc corpus: %d file(s), %d chars total.", len(doc_parts), len(all_doc_text))
        sys.stdout.flush()

    # Decide: inject everything into context, or build RAG indices?
    need_rag = False

    if full_rtl_context and len(all_doc_text) < config.context_injection_threshold:
        # Both RTL and docs fit in context — no RAG needed.
        if all_doc_text:
            full_docs_context = all_doc_text
            logger.info(
                "Injecting docs into context — skipping retriever build to save memory."
            )
        # else: no docs at all, just RTL context.
    else:
        # Corpus too large — need to build retrieval indices.
        need_rag = True

    # AST-only mode doesn't need retrieval — skip heavy index building.
    if need_rag and config.ast_only and config.use_ast_assertions:
        logger.info(
            "AST-only mode — skipping retrieval index build to save memory."
        )
        need_rag = False

    if need_rag:
        logger.info("Building retrieval indices (importing heavy libraries) …")
        sys.stdout.flush()

        from sva_pipeline.rag import (
            load_rtl_chunks, load_doc_chunks,
            build_or_load_retriever, build_or_load_hybrid_retriever,
            build_module_summary_chunks, build_or_load_hierarchical_retriever,
        )

        rtl_checksum = compute_source_checksum(config.rtl_dir)
        should_rebuild = (
            config.force_rebuild_index
            or is_index_stale(config.rtl_index_path, rtl_checksum)
        )

        # RTL chunks with hierarchy-enriched prefixes.
        rtl_chunks = load_rtl_chunks(
            config.rtl_dir, max_chars=config.rtl_max_chunk_chars,
            design_info=design_info,
        )

        # Choose RTL retriever: hierarchical (large designs) or flat.
        rtl_model = config.rtl_embedding_model
        num_modules = len(design_info.modules) if design_info else 0

        if (config.use_hierarchical_retrieval
                and num_modules >= 5
                and full_rtl_context is None):
            # Two-stage: module summaries → filtered code search.
            logger.info("Building hierarchical RTL retriever (%d modules)…", num_modules)
            summary_chunks = build_module_summary_chunks(design_info)
            rtl_retriever = build_or_load_hierarchical_retriever(
                summary_index_path=config.module_summary_index_path,
                code_index_path=config.rtl_index_path,
                summary_chunks=summary_chunks,
                code_chunks=rtl_chunks,
                rtl_embedding_model=rtl_model,
                stage1_k=config.hierarchical_stage1_k,
                stage2_k=config.hierarchical_stage2_k,
                use_hybrid=config.use_hybrid_retrieval,
                rrf_k=config.rrf_k,
                force_rebuild=should_rebuild,
            )
        elif config.use_hybrid_retrieval and full_rtl_context is None:
            rtl_retriever = build_or_load_hybrid_retriever(
                index_path=config.rtl_index_path,
                chunks=rtl_chunks,
                embedding_model=rtl_model,
                rrf_k=config.rrf_k,
                force_rebuild=should_rebuild,
                batch_size=config.rag_encoder_batch_size,
            )
        else:
            rtl_retriever = build_or_load_retriever(
                index_path=config.rtl_index_path,
                chunks=rtl_chunks,
                embedding_model=rtl_model,
                force_rebuild=should_rebuild,
                batch_size=config.rag_encoder_batch_size,
            )
        save_checksum(config.rtl_index_path, rtl_checksum)
        logger.info("RTL retriever complete — proceeding to doc retriever.")
        sys.stdout.flush()

        # Doc chunks — use NLP embedding model (separate from RTL).
        doc_model = config.doc_embedding_model
        doc_chunks = []
        logger.info("STEP A: doc_model=%s", doc_model); sys.stdout.flush()
        if config.docs_dir:
            logger.info("STEP B: loading doc chunks from %s", config.docs_dir); sys.stdout.flush()
            doc_chunks = load_doc_chunks(
                config.docs_dir,
                chunk_size=config.doc_chunk_size,
                overlap=config.doc_chunk_overlap,
            )
            logger.info("STEP C: loaded %d doc chunks", len(doc_chunks)); sys.stdout.flush()
        if config.threat_model and os.path.exists(config.threat_model):
            from sva_pipeline.rag import chunk_document_file
            doc_chunks.extend(chunk_document_file(
                config.threat_model,
                chunk_size=config.doc_chunk_size,
                overlap=config.doc_chunk_overlap,
            ))

        logger.info("STEP D: computing doc checksum"); sys.stdout.flush()
        doc_checksum = compute_source_checksum(
            config.docs_dir or ".", extensions={".md", ".txt"}
        ) if config.docs_dir else ""
        logger.info("STEP E: doc_checksum=%s", doc_checksum[:16] if doc_checksum else "(empty)"); sys.stdout.flush()
        doc_rebuild = should_rebuild or (
            doc_checksum and is_index_stale(config.doc_index_path, doc_checksum)
        )
        logger.info("STEP F: doc_rebuild=%s", doc_rebuild); sys.stdout.flush()

        logger.info("About to build doc retriever: %d chunks, model=%s", len(doc_chunks), doc_model)
        sys.stdout.flush()
        if config.use_hybrid_retrieval:
            doc_retriever = build_or_load_hybrid_retriever(
                index_path=config.doc_index_path,
                chunks=doc_chunks,
                embedding_model=doc_model,
                rrf_k=config.rrf_k,
                force_rebuild=doc_rebuild,
                batch_size=config.rag_encoder_batch_size,
            )
        else:
            doc_retriever = build_or_load_retriever(
                index_path=config.doc_index_path,
                chunks=doc_chunks,
                embedding_model=doc_model,
                force_rebuild=doc_rebuild,
                batch_size=config.rag_encoder_batch_size,
            )
        logger.info("Doc retriever complete.")
        sys.stdout.flush()
        if doc_checksum:
            save_checksum(config.doc_index_path, doc_checksum)

    # ---- Append injected docs to the RTL context if available -------------
    if full_docs_context and full_rtl_context:
        full_rtl_context += (
            "\n\n"
            "═══════════════════════════════════════════════════════════\n"
            "DESIGN DOCUMENTATION (read this to derive assertions)\n"
            "═══════════════════════════════════════════════════════════\n"
            + full_docs_context
        )
        logger.info("Docs injected into context alongside RTL.")

    # ---- Free embedding models from GPU before loading the LLM ------------
    import gc
    try:
        from sva_pipeline.rag import _ENCODER_CACHE
        for enc in _ENCODER_CACHE.values():
            enc.cpu()
        logger.info("Freed embedding model(s) from GPU.")
    except (ImportError, AttributeError):
        pass
    gc.collect()

    # ---- Extract structured RTL facts via pyslang ------------------------
    # Built once per run, consumed by both the agent (for prompt augmentation
    # when use_rtl_facts is enabled) and the post-processors. No caching —
    # fast enough on typical designs (<1s) and avoids staleness bugs.
    logger.info("Extracting RTL facts via pyslang …")
    from sva_pipeline.rtl_facts import extract_rtl_facts
    rtl_facts = extract_rtl_facts(
        config.rtl_dir,
        design_info.signal_map,
        top_module=config.top_module,
        module_facts_mode=getattr(config, "module_facts_mode", "off"),
        detect_out_of_scope=getattr(
            config, "detect_out_of_scope_structural", True,
        ),
        sink_module_patterns=(
            getattr(config, "out_of_scope_sink_module_patterns", None) or None
        ),
        out_of_scope_max_iterations=getattr(
            config, "out_of_scope_propagation_max_iterations", 10,
        ),
        docs_dir=getattr(config, "docs_dir", None),
    )

    # ---- Analysis exports (hierarchy, facts, AST, trace) -----------------
    analysis_tracer = None
    if getattr(config, "export_analysis", True):
        analysis_dir = resolve_analysis_dir(config)
        if ensure_dir(analysis_dir):
            logger.info("Analysis exports -> %s", analysis_dir)
            # Hierarchy + facts exports (need a Compilation object).
            try:
                import pyslang
                from pathlib import Path as _P
                _comp = pyslang.Compilation()
                for _r, _, _files in os.walk(config.rtl_dir):
                    for _fn in sorted(_files):
                        if _P(_fn).suffix in {".v", ".sv"}:
                            _t = pyslang.SyntaxTree.fromFile(
                                os.path.join(_r, _fn))
                            _comp.addSyntaxTree(_t)
                from sva_pipeline.rtl_facts import (
                    _is_sink_module, DEFAULT_SINK_MODULE_PATTERNS,
                )
                _patterns = (
                    getattr(config, "out_of_scope_sink_module_patterns", None)
                    or DEFAULT_SINK_MODULE_PATTERNS
                )
                _sink_check = (
                    lambda inst: _is_sink_module(inst, _patterns)
                )
                export_hierarchy(
                    analysis_dir, _comp, config.top_module, _sink_check,
                )
            except Exception as exc:
                logger.warning("Hierarchy export failed: %s", exc)
            export_rtl_facts_json(analysis_dir, rtl_facts)
            analysis_tracer = AssertionTracer(analysis_dir)
        else:
            logger.warning("Analysis export disabled (dir creation failed).")

    # ---- Instantiate and run the agent -----------------------------------
    # Import agent here (deferred) to avoid loading torch/transformers at startup.
    if config.ast_only and config.use_ast_assertions:
        logger.info("Initialising SVA agent (AST-only — no LLM loaded) …")
    else:
        logger.info("Initialising SVA agent (loading LLM — this uses ~16 GB GPU) …")
    sys.stdout.flush()
    from sva_pipeline.agent import SVAAgent
    agent = SVAAgent(
        config=config,
        rtl_retriever=rtl_retriever,
        doc_retriever=doc_retriever,
        hierarchy=design_info.hierarchy_text,
        signal_map=design_info.signal_map,
        design_graph=None,
        graph_summary_text=design_info.graph_summary_text,
        full_rtl_context=full_rtl_context,
        facts=rtl_facts,
    )
    # Pass clock/reset info for AST-guided assertion generation.
    agent._clock_signal = design_info.clock_signal
    agent._reset_signal = design_info.reset_signal
    # Thread the analysis tracer into the agent for AST/LLM step logging.
    agent._analysis_tracer = analysis_tracer

    logger.info("Running assertion generation …")
    raw_sva = agent.generate_assertions(config.task)

    # ---- Post-generation lint feedback loop ------------------------------
    logger.info("Starting lint feedback loop …")
    from sva_pipeline.lint_loop import run_lint_loop
    sva_code = run_lint_loop(
        agent, raw_sva, config,
        facts=rtl_facts,
        signal_map=design_info.signal_map,
        clock_signal=design_info.clock_signal,
        reset_signal=design_info.reset_signal,
        analysis_tracer=analysis_tracer,
    )

    # ---- Security pass (optional) ---------------------------------------
    # Runs only when config.security_pass_enabled and a threat_model file
    # is configured. Appends CWE-tagged assertions to sva_code. Token usage
    # is logged through agent.trace and rolls into the global summary.
    if getattr(config, "security_pass_enabled", False) and getattr(config, "threat_model", ""):
        logger.info("Running security pass …")
        from sva_pipeline.security_pass import run_security_pass
        sva_code = run_security_pass(
            agent, sva_code, config,
            facts=rtl_facts,
            signal_map=design_info.signal_map,
            clock_signal=design_info.clock_signal,
            reset_signal=design_info.reset_signal,
        )

    # ---- Write outputs ---------------------------------------------------
    if sva_code.strip():
        write_output(sva_code, config.output_sva_file, config.log_file)
        print("\n" + "=" * 70)
        print("GENERATED SVA ASSERTIONS")
        print("=" * 70)
        print(sva_code)
        print("=" * 70)
        print(f"\nOutput written to: {config.output_sva_file}")
        print(f"Lint report: {config.lint_failures_file}")
        _failed_run = False
    else:
        logger.error("No assertions survived — check %s", config.lint_failures_file)
        _failed_run = True
        # Don't sys.exit yet — fall through to save trace + token
        # summary (cost data is valuable even when the run failed).

    # ---- Per-module CSV + tracer cleanup ---------------------------------
    if analysis_tracer is not None:
        analysis_tracer.record(
            "3-final", "output_file", action="keep",
            delta=sva_code.count("assert"),
            description="Final SVA file",
        )
        analysis_tracer.close()
        try:
            export_per_module_csv(
                resolve_analysis_dir(config),
                config.rtl_dir,
                ast_skeletons=getattr(agent, "_ast_skeletons", None),
                final_sva_path=config.output_sva_file,
                facts=rtl_facts,
            )
            logger.info("Per-module assertion CSV written.")
        except Exception as exc:
            logger.warning("Per-module CSV export failed: %s", exc)

    # ---- Mutation testing (optional) -------------------------------------
    if config.mutation.enabled:
        logger.info("Starting mutation testing …")
        from sva_pipeline.mutation import run_mutation_testing
        mutation_report = run_mutation_testing(config, design_info)

        # Log mutation results to the trace.
        if hasattr(agent, 'trace'):
            meta = mutation_report.get("metadata", {})
            agent.trace.log_step(
                phase="mutation_testing",
                step=1,
                model_output="",
                notes=(
                    f"total={meta.get('total_mutants', 0)}, "
                    f"killed={meta.get('killed', 0)}, "
                    f"survived={meta.get('survived', 0)}, "
                    f"stillborn={meta.get('stillborn', 0)}, "
                    f"score={meta.get('mutation_score_pct', 'N/A')}"
                ),
            )

    # ---- Save the full pipeline trace ------------------------------------
    if hasattr(agent, 'trace'):
        agent.trace.save()
        logger.info("Pipeline trace saved.")

    # ---- Token-cost summary ----------------------------------------------
    # Sum token usage across every LLM call this run made. Also count the
    # final surviving assertions from the output SVA file so downstream
    # analysis can compute tokens-per-assertion. Written to
    # <sva_file>_token_summary.json alongside the trace files.
    if hasattr(agent, "trace"):
        try:
            import json as _json
            import re as _re

            totals = agent.trace.totals()
            final_asserts = 0
            try:
                with open(config.output_sva_file, "r", encoding="utf-8") as _f:
                    _content = _f.read()
                final_asserts = len(
                    _re.findall(r"^\s*assert\b", _content, flags=_re.MULTILINE)
                )
            except OSError:
                pass

            summary = {
                "config": yaml_path,
                "design": getattr(config, "top_module", ""),
                "rtl_dir": getattr(config, "rtl_dir", ""),
                "model": getattr(config, "model_id", ""),
                "backend": getattr(config, "model_backend", ""),
                "naive_baseline": bool(
                    getattr(config, "naive_baseline", False)
                ),
                "llm_calls": totals["llm_calls"],
                "prompt_tokens": totals["prompt_tokens"],
                "completion_tokens": totals["completion_tokens"],
                "total_tokens": totals["total_tokens"],
                "final_assertions": final_asserts,
                "tokens_per_assertion": (
                    round(totals["total_tokens"] / final_asserts, 2)
                    if final_asserts > 0 else None
                ),
            }

            sva_path = Path(config.output_sva_file)
            summary_path = sva_path.with_name(
                sva_path.stem + "_token_summary.json"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as _f:
                _json.dump(summary, _f, indent=2)
            logger.info(
                "Token usage: prompt=%d, completion=%d, total=%d across "
                "%d LLM call(s) -> %d assertion(s) "
                "(%.2f tokens/assertion). Summary: %s",
                totals["prompt_tokens"], totals["completion_tokens"],
                totals["total_tokens"], totals["llm_calls"],
                final_asserts,
                (totals["total_tokens"] / final_asserts)
                if final_asserts > 0 else 0.0,
                summary_path,
            )
        except Exception as exc:
            logger.debug("Token summary write failed: %s", exc)

    # If the lint loop produced no surviving assertions, exit non-zero now
    # (after the trace + token summary are written so cost data isn't lost).
    if _failed_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
