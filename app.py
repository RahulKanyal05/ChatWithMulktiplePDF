import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------
# Config / env
# -----------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "gemini-2.5-flash"  # you can switch to gemini-1.5-pro if you want
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DIR = "faiss_index"


# -----------------------
# Helpers
# -----------------------
def ensure_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "pdf_texts" not in st.session_state:
        st.session_state.pdf_texts = {}          # {filename: text}
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}          # {filename: summary}
    if "faiss_built" not in st.session_state:
        st.session_state.faiss_built = False


def extract_text_from_pdf(uploaded_file):
    """Return extracted text from a single PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        texts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                texts.append(t)
        return "\n".join(texts)
    except Exception as e:
        st.error(f"Failed to read {uploaded_file.name}: {e}")
        return ""


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def build_faiss_index(uploaded_files):
    """Extract text, split to chunks, build FAISS with local embeddings."""
    all_texts = []
    all_metadatas = []

    for f in uploaded_files:
        name = f.name
        if name not in st.session_state.pdf_texts:
            st.session_state.pdf_texts[name] = extract_text_from_pdf(f)

        chunks = get_text_chunks(st.session_state.pdf_texts[name])
        all_texts.extend(chunks)
        all_metadatas.extend([{"source": name}] * len(chunks))

    if not all_texts:
        raise ValueError("No text extracted from PDFs.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = FAISS.from_texts(all_texts, embedding=embeddings, metadatas=all_metadatas)
    vs.save_local(FAISS_DIR)
    st.session_state.faiss_built = True


def load_faiss_index():
    """Load FAISS index from disk."""
    if not os.path.exists(FAISS_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)


def summarize_with_llm(text: str) -> str:
    """Summarize text using Gemini directly, no chains."""
    if not GOOGLE_API_KEY:
        return "ERROR: GOOGLE_API_KEY is not set. Cannot summarize."

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.6)
    prompt = f"""
Provide a detailed and comprehensive summary of the following text.
Focus on key points and important details.

Text:
{text}

Detailed Summary:
"""
    resp = llm.invoke(prompt)
    # resp is an AIMessage; .content is usually the text
    try:
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        return str(resp)


def answer_question_from_docs(docs, question: str) -> str:
    """Use retrieved docs + Gemini to answer a question. No chains."""
    if not GOOGLE_API_KEY:
        return "ERROR: GOOGLE_API_KEY is not set. Cannot answer questions."

    context_parts = []
    for d in docs:
        # d is a LangChain Document
        context_parts.append(d.page_content)

    context = "\n\n".join(context_parts)

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not present in the context, say exactly:
"Answer is not available in the context."

Context:
{context}

Question: {question}

Answer:
"""
    resp = llm.invoke(prompt)
    try:
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        return str(resp)


def get_similarity_docs(query: str, k: int = 4):
    store = load_faiss_index()
    if store is None:
        return []
    return store.similarity_search(query, k=k)


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
st.title("Chat with Multiple PDFs — Local Embeddings + Gemini")

ensure_session_state()

# Sidebar: upload & index
with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        st.session_state.uploaded_files = uploaded

    st.write("---")
    if st.button("Process & Build Index"):
        if not st.session_state.uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            try:
                with st.spinner("Extracting text and building FAISS index..."):
                    build_faiss_index(st.session_state.uploaded_files)
                st.success("Index built successfully.")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

# Main layout
st.subheader("Summarize Selected PDFs")

uploaded_names = [f.name for f in st.session_state.uploaded_files] if st.session_state.uploaded_files else []
if uploaded_names:
    st.info("Uploaded files: " + ", ".join(uploaded_names))

names_input = st.text_input(
    "Enter filenames to summarize (comma-separated)",
    placeholder="example1.pdf, example2.pdf"
)

if st.button("Summarize"):
    if not uploaded_names:
        st.error("No PDFs uploaded.")
    else:
        requested = [n.strip() for n in names_input.split(",") if n.strip()]
        if not requested:
            st.warning("Enter at least one filename.")
        else:
            invalid = [n for n in requested if n not in uploaded_names]
            if invalid:
                st.error("These filenames are not uploaded: " + ", ".join(invalid))
            else:
                for name in requested:
                    if name not in st.session_state.pdf_texts:
                        file_obj = next((f for f in st.session_state.uploaded_files if f.name == name), None)
                        if file_obj is None:
                            st.error(f"Could not find uploaded file object for {name}")
                            continue
                        st.session_state.pdf_texts[name] = extract_text_from_pdf(file_obj)

                    if name not in st.session_state.summaries:
                        with st.spinner(f"Summarizing {name}..."):
                            summary = summarize_with_llm(st.session_state.pdf_texts[name])
                            st.session_state.summaries[name] = summary

                    st.markdown(f"**Summary — {name}**")
                    st.write(st.session_state.summaries[name])

st.write("---")
st.subheader("Ask a Question Across PDFs")

user_q = st.text_input("Your question")

if st.button("Get Answer"):
    if not user_q:
        st.warning("Type a question first.")
    elif not st.session_state.faiss_built:
        st.error("FAISS index not built yet. Upload PDFs and click 'Process & Build Index'.")
    else:
        with st.spinner("Searching and generating answer..."):
            docs = get_similarity_docs(user_q, k=4)
            if not docs:
                st.error("No relevant documents found. Try rebuilding the index.")
            else:
                answer = answer_question_from_docs(docs, user_q)
                st.markdown("**Answer:**")
                st.write(answer)

                # Show sources
                sources = {d.metadata.get("source", "unknown") for d in docs}
                st.caption("Sources: " + ", ".join(sources))

