#!/bin/sh

set -eu

BM25_INDEX_FILE="/app/data/bm25_index/corpus.pkl"
CHROMA_INDEX_DIR="/app/data/chroma_db"
RAW_DATA_FILE="/app/data/raw/aapl_10k.json"

log() {
    printf '[startup] %s\n' "$1"
}

has_chroma_index() {
    if [ ! -d "$CHROMA_INDEX_DIR" ]; then
        return 1
    fi

    if find "$CHROMA_INDEX_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi

    return 1
}

ensure_index() {
    if [ -f "$BM25_INDEX_FILE" ] && has_chroma_index; then
        log "Existing indexes detected, skip rebuild."
        return 0
    fi

    if [ ! -f "$RAW_DATA_FILE" ]; then
        log "Missing source data: $RAW_DATA_FILE"
        return 1
    fi

    log "Index not found, running scripts/build_index.py ..."
    python3 scripts/build_index.py
    log "Index build completed."
}

shutdown() {
    log "Stopping services..."
    kill "$api_pid" "$ui_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
    wait "$ui_pid" 2>/dev/null || true
}

ensure_index

log "Starting FastAPI on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
uvicorn app.api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" &
api_pid=$!

log "Starting Streamlit on ${STREAMLIT_HOST:-0.0.0.0}:${STREAMLIT_PORT:-8501}"
streamlit run ui/streamlit_app.py \
    --server.address "${STREAMLIT_HOST:-0.0.0.0}" \
    --server.port "${STREAMLIT_PORT:-8501}" &
ui_pid=$!

trap shutdown INT TERM

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$ui_pid" 2>/dev/null; do
    sleep 1
done

log "One service exited unexpectedly."
shutdown
exit 1
