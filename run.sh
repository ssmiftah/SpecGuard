#!/usr/bin/env bash
# Generic runner for the SVA generation pipeline.
# Usage: ./run.sh <config.yaml>

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument check
# ---------------------------------------------------------------------------
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <config.yaml>"
    echo ""
    echo "Runs the SVA pipeline with the given YAML config file."
    echo "Stdout and stderr are teed to a log file alongside the config."
    exit 0
fi

CONFIG="$1"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Derive a run-log path from the config filename
# e.g. examples/coral_fifox.yaml -> run_coral_fifox.log
# ---------------------------------------------------------------------------
BASENAME="$(basename "$CONFIG" .yaml)"
RUNLOG="run_${BASENAME}.log"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
SEP="========================================"
echo "$SEP"
echo " SVA Pipeline — $BASENAME"
echo " Config : $CONFIG"
echo " Run log: $RUNLOG"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "$SEP"
echo ""

# ---------------------------------------------------------------------------
# VRAM monitor — polls nvidia-smi every 2 s in the background, records
# per-GPU used memory (MiB) to a sidecar log; peak is reported at the end.
# ---------------------------------------------------------------------------
VRAMLOG="run_${BASENAME}_vram.log"
VRAM_PID=""

if command -v nvidia-smi &>/dev/null; then
    # Write a header then append one timestamped sample every 2 seconds.
    echo "timestamp,gpu,used_mib,total_mib" > "$VRAMLOG"
    (
        while true; do
            nvidia-smi \
                --query-gpu=index,memory.used,memory.total \
                --format=csv,noheader,nounits \
            | awk -v ts="$(date '+%H:%M:%S')" \
                '{printf "%s,%s,%s,%s\n", ts, $1, $2, $3}' \
            >> "$VRAMLOG"
            sleep 2
        done
    ) &
    VRAM_PID=$!
    echo " VRAM log: $VRAMLOG  (monitor PID $VRAM_PID)"
else
    echo " (nvidia-smi not found — VRAM monitoring disabled)"
fi
echo ""

# ---------------------------------------------------------------------------
# Run — tee stdout to RUNLOG; stderr goes to terminal AND the same log
# ---------------------------------------------------------------------------
set +e
python -u main.py "$CONFIG" \
    2>&1 | tee "$RUNLOG"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

# Stop the VRAM monitor as soon as the pipeline exits.
if [[ -n "$VRAM_PID" ]]; then
    kill "$VRAM_PID" 2>/dev/null && wait "$VRAM_PID" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
echo ""
echo "$SEP"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Exit code: $EXIT_CODE"
echo "$SEP"

# ---------------------------------------------------------------------------
# Tail the run log (last 30 lines)
# ---------------------------------------------------------------------------
echo ""
echo "--- Last 30 lines of $RUNLOG ---"
tail -n 30 "$RUNLOG"

# ---------------------------------------------------------------------------
# VRAM summary — peak used MiB per GPU over the run
# ---------------------------------------------------------------------------
if [[ -f "$VRAMLOG" ]] && [[ "$(wc -l < "$VRAMLOG")" -gt 1 ]]; then
    echo ""
    echo "--- VRAM peak usage (from $VRAMLOG) ---"
    # For each GPU index, find the maximum used_mib recorded.
    awk -F',' 'NR>1 { gpu=$2; used=$3; total=$4
                      gsub(/ /,"",gpu); gsub(/ /,"",used); gsub(/ /,"",total)
                      if (used+0 > peak[gpu]+0) { peak[gpu]=used; tot[gpu]=total } }
               END   { for (g in peak)
                           printf "  GPU %s : %s / %s MiB (%.1f%%)\n",
                               g, peak[g], tot[g], (peak[g]+0)/(tot[g]+0)*100 }' \
        "$VRAMLOG" | sort
fi

# ---------------------------------------------------------------------------
# Parse SVA output path from YAML and show assertion stats
# ---------------------------------------------------------------------------
SVA_FILE=""
if command -v grep &>/dev/null; then
    # Extract "sva_file:" value — handles both quoted and unquoted paths
    SVA_FILE=$(grep -E '^\s*sva_file\s*:' "$CONFIG" \
               | head -1 \
               | sed 's/.*sva_file\s*:\s*["'"'"']*//' \
               | sed 's/["'"'"']*$//' \
               | tr -d '[:space:]')
fi

if [[ -n "$SVA_FILE" && -f "$SVA_FILE" ]]; then
    echo ""
    echo "--- Assertion summary: $SVA_FILE ---"
    TOTAL=$(grep -cE '^\s*assert\b' "$SVA_FILE" 2>/dev/null || echo 0)
    CONCURRENT=$(grep -cE '^\s*assert property\b' "$SVA_FILE" 2>/dev/null || echo 0)
    IMMEDIATE=$(( TOTAL - CONCURRENT ))
    echo "  Total assertions  : $TOTAL"
    echo "  Concurrent        : $CONCURRENT"
    echo "  Immediate         : $IMMEDIATE"
    echo "  \$past usage       : $(grep -c '\$past' "$SVA_FILE" 2>/dev/null || echo 0)"
    echo "  |-> usage         : $(grep -c '|->' "$SVA_FILE" 2>/dev/null || echo 0)"
    echo "  Reset assertions  : $(grep -ci 'reset' "$SVA_FILE" 2>/dev/null || echo 0)"
else
    if [[ -n "$SVA_FILE" ]]; then
        echo ""
        echo "  (SVA output file not found: $SVA_FILE)"
    fi
fi

# ---------------------------------------------------------------------------
# Key pipeline events extracted from the run log
# ---------------------------------------------------------------------------
echo ""
echo "--- Pipeline events ---"
grep -E \
    "AST: excluded|Skipped .* trivial|Out-of-scope|Lint summary|\
fixed [0-9]|removed [0-9]|Deduplicat|Split into|\
Token usage|final output|No assertions survived|ERROR" \
    "$RUNLOG" 2>/dev/null | tail -20 || true

echo ""
exit "$EXIT_CODE"
