# Repository Guidelines

## Project Structure & Module Organization
`app/` contains the core RAG pipeline: `ingest/` parses and chunks 10-K data, `retrieve/` implements BM25, Chroma, and hybrid retrieval, `generate/` builds answers, `api/` exposes FastAPI endpoints, and `core/` stores settings. `ui/streamlit_app.py` is the main local interface. `scripts/` holds operational helpers such as `setup.py`, `build_index.py`, `debug_retrieve.py`, and `debug_answer.py`. Repository data lives under `data/` (`raw/`, `processed/`, `chroma_db/`, `bm25_index/`); generated indexes also use `indexes/` and `embeddings/`. `tests/` exists but currently has no committed automated tests.

## Build, Test, and Development Commands
Set up a local environment with `python3 -m venv venv && source venv/bin/activate`, then install dependencies via `pip install -r requirements.txt`. Use `python3 scripts/setup.py` for the full bootstrap flow, including NLTK downloads and required directories. Build retrieval indexes with `python3 scripts/build_index.py`. Run the UI locally with `streamlit run ui/streamlit_app.py`. Start the API with `python3 -m app.api.main` and use `docker-compose up -d` for containerized development.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and follow existing module boundaries; prefer small, reusable functions over new abstractions. Keep changes minimal and avoid touching unrelated modules. Use `snake_case` for modules, functions, variables, and config keys; use `PascalCase` for Pydantic models and service classes. Write comments in English, keep them sparse, and favor clear code over explanatory noise. No formatter or linter is configured here, so match the surrounding style before committing.

## Testing Guidelines
There is no stable automated test suite yet, so validate changes with targeted manual checks. For retrieval changes, run `python3 scripts/debug_retrieve.py "apple cash flow" --mode hybrid`. For answer generation, run `python3 scripts/debug_answer.py "苹果现金流情况如何？" --show-retrieved`. Add new automated tests under `tests/` with `test_*.py` naming when introducing behavior that can be isolated.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit prefixes such as `feat:`, `fix:`, and `docs:`. Keep commit messages one line, imperative, and under 72 characters, for example `fix: handle empty chroma results`. PRs should describe the user-visible change, list validation steps, link related issues or plans, and include UI screenshots when `ui/` or Streamlit behavior changes.

## Configuration & Data Notes
Copy `.env.example` to `.env` before local runs. Keep secrets such as `DEEPSEEK_API_KEY` out of git. Large generated artifacts in `data/`, `indexes/`, and `embeddings/` should be treated as build outputs unless the change explicitly targets sample data.
