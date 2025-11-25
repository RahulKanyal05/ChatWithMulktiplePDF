import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------
# Config / env
# -----------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # required only for LLM (summaries/answers)
# You can still use the app without this key; only summarization/LLM answers will warn/fail.

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# LLM model (for summary + QA). You can change to "gemini-2.5-pro" if available to you.
LLM_MODEL = "gemini-2.5-pro"

# Local embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Path for FAISS index storage
FAISS_DIR = "faiss_index"

# -----------------------
# Helpers
# -----------------------
def ensure_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []          # list of UploadedFile
    if "pdf_texts" not in st.session_state:
        st.session_state.pdf_texts = {}              # {filename: full_text}
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}              # {filename: summary}
    if "faiss_built" not in st.session_state:
        st.session_state.faiss_built = False

def extract_text_from_pdf(uploaded_file):
    """Return text extracted from a single PyPDF2 PdfReader file-like object."""
    try:
        pdf_reader = PdfReader(uploaded_file)
        pages = []
        for p in pdf_reader.pages:
            page_text = p.extract_text()
            if page_text:
                pages.append(page_text)
        return "\n".join(pages)
    except Exception as e:
        st.error(f"Failed to read {uploaded_file.name}: {e}")
        return ""

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_faiss_index(chunks, store_path=FAISS_DIR):
    """Build and save a FAISS index from a list of chunk strings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    # chunks are plain strings; we save filename meta in the chunk text using a separator
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    vs.save_local(store_path)
    st.session_state.faiss_built = True
    return vs

def load_faiss_index(store_path=FAISS_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    if not os.path.exists(store_path):
        return None
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

def get_qa_chain():
    """Return a QA chain using Gemini LLM. Requires GOOGLE_API_KEY."""
    prompt_template = """
Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
say exactly: "Answer is not available in the context." Do not hallucinate.

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def summarize_with_llm(text):
    """Summarize text using Gemini LLM. Returns string or error note."""
    if not GOOGLE_API_KEY:
        return "ERROR: GOOGLE_API_KEY not set — cannot summarize with Gemini."
    summary_prompt = """
Provide a detailed and comprehensive summary of the following text. Include key points and important details.

Text:
{context}

Detailed Summary:
"""
    prompt = PromptTemplate(template=summary_prompt, input_variables=["context"])
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.6)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    res = chain({"input_documents": [Document(page_content=text)]}, return_only_outputs=True)
    # langchain returns "output_text" or "answer" depending on version
    return res.get("output_text") or res.get("answer") or str(res)

def get_similarity_docs(query, k=4):
    store = load_faiss_index()
    if store is None:
        return []
    results = store.similarity_search(query, k=k)
    # results may be Document objects or plain strings depending on version
    doc_objs = []
    for r in results:
        content = r.page_content if hasattr(r, "page_content") else r
        # we used "FILENAME||SEP||chunk" when building; split if present
        if "||SEP||" in content:
            fname, chunk = content.split("||SEP||", 1)
            doc_objs.append(Document(page_content=chunk, metadata={"source": fname}))
        else:
            doc_objs.append(Document(page_content=content, metadata={"source": "unknown"}))
    return doc_objs

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
st.title("Chat with Multiple PDFs — Local Embeddings + FAISS")

ensure_session_state()

# Sidebar: upload & process
with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Upload PDF files (multiple)", type=["pdf"], accept_multiple_files=True)

    if uploaded:
        # store the list of uploaded Files (Streamlit UploadedFile) in session_state
        st.session_state.uploaded_files = uploaded

    st.write("---")
    if st.button("Process & Build Index"):
        if not st.session_state.uploaded_files:
            st.warning("Please upload at least one PDF file before processing.")
        else:
            st.info("Extracting text from PDFs...")
            all_chunks = []
            for f in st.session_state.uploaded_files:
                name = f.name
                if name not in st.session_state.pdf_texts:
                    txt = extract_text_from_pdf(f)
                    st.session_state.pdf_texts[name] = txt
                # split into chunks and add filename metadata
                chunks = get_text_chunks(st.session_state.pdf_texts[name])
                chunks_with_meta = [f"{name}||SEP||{c}" for c in chunks]
                all_chunks.extend(chunks_with_meta)

            if not all_chunks:
                st.error("No text extracted from uploaded PDFs.")
            else:
                try:
                    with st.spinner("Creating embeddings and building FAISS index (local)..."):
                        build_faiss_index(all_chunks)
                    st.success("FAISS index built and saved locally.")
                except Exception as e:
                    st.error(f"Failed to build FAISS index: {e}")

# Main columns
left = st.container()

with left:
    st.subheader("Summarize Selected PDFs")
    st.write("Enter filenames exactly as uploaded (comma-separated).")

    uploaded_names = [f.name for f in st.session_state.uploaded_files] if st.session_state.uploaded_files else []
    if uploaded_names:
        st.info("Uploaded files: " + ", ".join(uploaded_names))

    names_input = st.text_input("PDF filenames to summarize (comma-separated)", placeholder="example.pdf, another.pdf")
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
                    st.error(f"These filenames are not uploaded: {', '.join(invalid)}")
                else:
                    # extract texts if missing
                    for name in requested:
                        if name not in st.session_state.pdf_texts:
                            file_obj = next((f for f in st.session_state.uploaded_files if f.name == name), None)
                            if file_obj:
                                st.session_state.pdf_texts[name] = extract_text_from_pdf(file_obj)
                            else:
                                st.error(f"Could not find object for {name}")
                                continue

                        # Summarize if not cached
                        if name not in st.session_state.summaries:
                            with st.spinner(f"Summarizing {name}..."):
                                s = summarize_with_llm(st.session_state.pdf_texts[name])
                                st.session_state.summaries[name] = s

                        st.markdown(f"**Summary — {name}**")
                        st.write(st.session_state.summaries[name])

    st.write("---")
    st.subheader("Chat / Ask a Question")
    user_q = st.text_input("Ask a question across uploaded PDFs (retrieval + LLM)")
    if st.button("Get Answer"):
        if not user_q:
            st.warning("Type a question first.")
        else:
            if not st.session_state.faiss_built:
                st.error("FAISS index not found. Upload PDFs and click 'Process & Build Index' first.")
            else:
                with st.spinner("Searching and generating answer..."):
                    docs = get_similarity_docs(user_q, k=4)
                    if not docs:
                        st.error("No relevant docs found or FAISS index missing. Rebuild index.")
                    else:
                        if not GOOGLE_API_KEY:
                            st.error("GOOGLE_API_KEY not set — cannot use Gemini for answers. Set it in environment.")
                        else:
                            try:
                                chain = get_qa_chain()
                                res = chain({"input_documents": docs, "question": user_q}, return_only_outputs=True)
                                answer = res.get("output_text") or res.get("answer") or str(res)
                                st.markdown("**Answer:**")
                                st.write(answer)
                                sources = list({d.metadata.get("source", "unknown") for d in docs})
                                st.caption("Sources: " + ", ".join(sources))
                            except Exception as e:
                                st.error(f"Error during LLM generation: {e}")
