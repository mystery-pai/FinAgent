"""
Streamlit UI for Fin-Agent AAPL 10-K Q&A System
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

# Configure page
st.set_page_config(
    page_title="Fin-Agent - AAPL 10-K Q&A",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .debug-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def load_system_components():
    """Load retriever and generator"""
    try:
        from app.retrieve.hybrid_retriever import HybridRetriever
        from app.generate.answer_generator import AnswerGenerator
        from app.core.config import settings

        # Initialize retriever
        retriever = HybridRetriever(
            bm25_index_path="indexes/bm25_index",
            chroma_collection_name="aapl_10k",
            chroma_persist_dir="indexes/chroma"
        )

        # Initialize generator
        generator = AnswerGenerator()

        return retriever, generator
    except Exception as e:
        st.error(f"Failed to load system components: {str(e)}")
        logger.error(f"Error loading components: {e}")
        return None, None


def render_header():
    """Render application header"""
    st.markdown('<div class="main-header">📊 Fin-Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AAPL 10-K 智能问答系统 | Intelligent Financial Q&A</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Retrieval settings
        st.subheader("Retrieval")
        top_k = st.slider("Top K Results", min_value=3, max_value=10, value=5)
        use_hybrid = st.checkbox("Use Hybrid Retrieval", value=True)

        # Year filter
        st.subheader("Filters")
        year_filter = st.selectbox(
            "Year Filter",
            options=[None, 2025, 2024, 2023, 2022, 2021, 2020],
            format_func=lambda x: "All Years" if x is None else str(x)
        )

        # LLM settings
        st.subheader("LLM")
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["deepseek", "ollama"],
            index=0
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Fin-Agent** is a RAG-based Q&A system for AAPL 10-K reports.

        - **Data**: AAPL 10-K (2020-2025)
        - **Retrieval**: Hybrid (BM25 + Vector)
        - **LLM**: DeepSeek / Ollama
        """)

        return top_k, use_hybrid, year_filter, llm_provider


def render_answer(answer: str, citations: list, debug_info: dict):
    """Render answer with citations and debug info"""
    # Answer
    st.markdown("### 📝 Answer")
    st.markdown(answer)

    # Citations
    if citations:
        st.markdown("### 📚 Citations")
        for i, citation in enumerate(citations, 1):
            with st.expander(f"Citation {i}: {citation.get('year', 'N/A')} - {citation.get('section_title', 'N/A')}"):
                st.markdown(f"**Chunk ID**: `{citation.get('chunk_id', 'N/A')}`")
                st.markdown(f"**Score**: `{citation.get('score', 0):.4f}`")
                st.markdown(f"**Item Type**: `{citation.get('item_type', 'N/A')}`")
                st.markdown("**Snippet**:")
                st.markdown(f"<div class='citation-box'>{citation.get('text', '')[:500]}...</div>", unsafe_allow_html=True)

    # Debug info
    with st.expander("🔍 Retrieval Debug Info"):
        st.markdown(f"<div class='debug-box'>", unsafe_allow_html=True)
        st.json(debug_info)
        st.markdown(f"</div>", unsafe_allow_html=True)


def render_query_history():
    """Render query history in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📜 Query History")

        if st.session_state.query_history:
            for i, query in enumerate(reversed(st.session_state.query_history), 1):
                if st.button(f"{i}. {query[:30]}...", key=f"history_{i}"):
                    st.session_state.query_input = query
                    st.rerun()
        else:
            st.markdown("*No queries yet*")


def main():
    """Main application"""
    # Initialize session state
    init_session_state()

    # Load components
    if st.session_state.retriever is None:
        with st.spinner("Loading system components..."):
            st.session_state.retriever, st.session_state.generator = load_system_components()

    if st.session_state.retriever is None:
        st.error("Failed to initialize system. Please check your configuration.")
        st.stop()

    # Render UI
    render_header()
    top_k, use_hybrid, year_filter, llm_provider = render_sidebar()

    # Main query interface
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Ask a question about AAPL 10-K...",
            placeholder="e.g., What were Apple's main risks in 2025?",
            key="query_input"
        )

    with col2:
        submitted = st.button("Search", type="primary", use_container_width=True)

    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        example_col1, example_col2, example_col3 = st.columns(3)

        with example_col1:
            if st.button("2025 Risks", use_container_width=True):
                st.session_state.query_input = "What were Apple's main risks in 2025?"
                st.rerun()
            if st.button("Revenue Trend", use_container_width=True):
                st.session_state.query_input = "How did Apple's revenue change from 2023 to 2025?"
                st.rerun()

        with example_col2:
            if st.button("Business Overview", use_container_width=True):
                st.session_state.query_input = "Summarize Apple's business overview"
                st.rerun()
            if st.button("Financial Health", use_container_width=True):
                st.session_state.query_input = "What is Apple's financial condition in 2025?"
                st.rerun()

        with example_col3:
            if st.button("Legal Issues", use_container_width=True):
                st.session_state.query_input = "What legal proceedings does Apple face?"
                st.rerun()
            if st.button("R&D Investment", use_container_width=True):
                st.session_state.query_input = "How much did Apple invest in R&D in 2024?"
                st.rerun()

    # Process query
    if submitted and query:
        # Add to history
        st.session_state.query_history.append(query)

        # Show progress
        with st.spinner("Retrieving relevant information..."):
            try:
                # Retrieve
                results = st.session_state.retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    year_filter=year_filter,
                    use_hybrid=use_hybrid
                )

                if not results:
                    st.warning("No relevant information found. Try rephrasing your query.")
                    st.stop()

                # Generate answer
                with st.spinner("Generating answer..."):
                    response = st.session_state.generator.generate(
                        query=query,
                        context_chunks=results,
                        llm_provider=llm_provider
                    )

                    # Display results
                    render_answer(
                        answer=response["answer"],
                        citations=response["citations"],
                        debug_info={
                            "query": query,
                            "top_k": top_k,
                            "use_hybrid": use_hybrid,
                            "year_filter": year_filter,
                            "retrieved_count": len(results),
                            "retrieval_method": "hybrid" if use_hybrid else "bm25"
                        }
                    )

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Query processing error: {e}")

    # Render query history
    render_query_history()


if __name__ == "__main__":
    main()
