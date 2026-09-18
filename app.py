import streamlit as st

from rag import (
    extract_text,
    create_chunks,
    create_vector_database,
    process_question
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "📚 Research Paper Question Answering System"
)

st.write(
    "Upload a research paper and ask questions "
    "about its contents using RAG."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ RAG Settings")

    chunk_size = st.number_input(
        "Chunk Size (words)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50
    )

    overlap = st.number_input(
        "Chunk Overlap (words)",
        min_value=0,
        max_value=300,
        value=100,
        step=10
    )

    top_k = st.number_input(
        "Retrieval Depth (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    st.markdown("---")

    st.info(
        "Recommended settings:\n\n"
        "Chunk Size: 500 words\n\n"
        "Overlap: 100 words\n\n"
        "Top-K: 5"
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "collection" not in st.session_state:

    st.session_state.collection = None


if "paper_processed" not in st.session_state:

    st.session_state.paper_processed = False


if "paper_name" not in st.session_state:

    st.session_state.paper_name = ""


if "pages_count" not in st.session_state:

    st.session_state.pages_count = 0


if "chunks_count" not in st.session_state:

    st.session_state.chunks_count = 0


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.subheader("1️⃣ Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    if st.button(
        "🔄 Process Research Paper",
        use_container_width=True
    ):

        with st.spinner(
            "Processing research paper..."
        ):

            try:

                # Read PDF
                pdf_bytes = uploaded_file.getvalue()

                # ----------------------------------
                # Extract text
                # ----------------------------------

                pages = extract_text(
                    pdf_bytes
                )

                if not pages:

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                    st.stop()

                # ----------------------------------
                # Create chunks
                # ----------------------------------

                chunks = create_chunks(
                    pages,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap)
                )

                if not chunks:

                    st.error(
                        "No text chunks could be created."
                    )

                    st.stop()

                # ----------------------------------
                # Create vector database
                # ----------------------------------

                client, collection = (
                    create_vector_database(
                        chunks
                    )
                )

                # ----------------------------------
                # Save in session state
                # ----------------------------------

                st.session_state.collection = (
                    collection
                )

                st.session_state.paper_processed = (
                    True
                )

                st.session_state.paper_name = (
                    uploaded_file.name
                )

                st.session_state.pages_count = (
                    len(pages)
                )

                st.session_state.chunks_count = (
                    len(chunks)
                )

                st.success(
                    "Research paper processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {str(e)}"
                )


# --------------------------------------------------
# PAPER INFORMATION
# --------------------------------------------------

if st.session_state.paper_processed:

    st.markdown("---")

    st.subheader(
        "📄 Uploaded Paper"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Paper",
            st.session_state.paper_name
        )

    with col2:

        st.metric(
            "Pages",
            st.session_state.pages_count
        )

    with col3:

        st.metric(
            "Chunks",
            st.session_state.chunks_count
        )


# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

st.markdown("---")

st.subheader(
    "2️⃣ Ask a Question"
)


question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What methodology was used?"
    )
)


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

if st.button(
    "🔍 Ask Question",
    use_container_width=True
):

    if not st.session_state.paper_processed:

        st.warning(
            "Please upload and process a "
            "research paper first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the paper and generating answer..."
        ):

            try:

                answer, pages, chunks = (
                    process_question(
                        st.session_state.collection,
                        question,
                        top_k=int(top_k)
                    )
                )

                # ----------------------------------
                # Display Answer
                # ----------------------------------

                st.markdown("---")

                st.subheader(
                    "💡 Answer"
                )

                st.write(answer)

                # ----------------------------------
                # Display Sources
                # ----------------------------------

                st.subheader(
                    "📑 Sources"
                )

                if pages:

                    source_text = ", ".join(
                        [
                            f"Page {page}"
                            for page in pages
                        ]
                    )

                    st.success(
                        source_text
                    )

                else:

                    st.write(
                        "No source pages found."
                    )

                # ----------------------------------
                # Retrieved Context
                # ----------------------------------

                with st.expander(
                    "🔎 View Retrieved Context"
                ):

                    for i, chunk in enumerate(
                        chunks
                    ):

                        st.markdown(
                            f"### Retrieved Chunk {i + 1}"
                        )

                        st.write(
                            f"**Page:** "
                            f"{chunk['page']}"
                        )

                        st.write(
                            chunk["text"]
                        )

                        st.markdown("---")

            except Exception as e:

                st.error(
                    f"Error answering question: {str(e)}"
                )


# --------------------------------------------------
# CLEAR BUTTON
# --------------------------------------------------

st.markdown("---")

if st.button(
    "🗑️ Clear Paper",
    use_container_width=True
):

    st.session_state.collection = None

    st.session_state.paper_processed = False

    st.session_state.paper_name = ""

    st.session_state.pages_count = 0

    st.session_state.chunks_count = 0

    st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Research Paper Question Answering System "
    "using Retrieval-Augmented Generation (RAG)"
)