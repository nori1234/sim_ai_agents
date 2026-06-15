#!/usr/bin/env bash
# Minimal local run of Emergence World on your own machine via Ollama.
#
#   1. Install Ollama: https://ollama.com  (then `ollama serve` in another terminal)
#   2. ./run_local.sh
#
# Defaults are tiny so it finishes in a few minutes on CPU (no GPU needed).
# Override anything with env vars, e.g.:
#   LLM_MODEL=qwen2.5:1.5b AGENTS=4 DAYS=2 ./run_local.sh   # even lighter
#   LLM_MODEL=llama3.1:8b  DAYS=15 ./run_local.sh           # heavier / longer
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${LLM_MODEL:-llama3.2:3b}"     # ~2GB, CPU-friendly, decent at JSON
PERSONA="${PERSONA:-claude}"
AGENTS="${AGENTS:-5}"
DAYS="${DAYS:-3}"
TICKS="${TICKS:-3}"
DB="${MEMORY_DB:-town.db}"            # persisted -> next run remembers this one
OLLAMA="${LLM_BASE_URL:-http://localhost:11434}"

echo "== Emergence World — minimal local run =="
echo "model=$MODEL  persona=$PERSONA  agents=$AGENTS  days=$DAYS  ticks=$TICKS"
echo

# 1) Ollama reachable?
if ! curl -sf "${OLLAMA%/v1}/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama not reachable at ${OLLAMA%/v1}." >&2
  echo "Start it in another terminal:  ollama serve" >&2
  exit 1
fi

# 2) Model present? pull if not.
if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
  echo "Pulling $MODEL (first time only)..."
  ollama pull "$MODEL"
fi

# 3) Run, grounded on memory + a changing environment.
export LLM_BASE_URL="${OLLAMA%/v1}/v1" LLM_MODEL="$MODEL" LLM_API_KEY="ollama"
python3 -m emergence.cli \
  --persona "$PERSONA" --llm --memory --environment \
  --memory-db "$DB" --agents "$AGENTS" --days "$DAYS" --ticks "$TICKS" --verbose

echo
echo "Memories persisted to $DB — run again and the town remembers this life."
