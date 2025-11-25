# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# # Check if API key is available
# if not api_key:
#     st.error("API Key not found. Please set your GOOGLE_API_KEY in the environment variables.")
#     st.stop()  # Stop execution if API key is missing

# genai.configure(api_key=api_key)

# # Function to extract text from uploaded PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Function to create and save a vector store from text chunks
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Function to create a conversational chain with a custom prompt
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
#     just say, "Answer is not available in the context," and do not provide a wrong answer.

#     Context: {context}

#     Question: {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Function to summarize text using the language model
# def summarize_text(text):
#     summarizer = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)  # Adjusted temperature for more variety
#     summary_prompt = """
#     Provide a detailed and comprehensive summary of the following text. Make sure to include all important points and key details:

#     Text: {context}

#     Detailed Summary:
#     """
#     prompt = PromptTemplate(template=summary_prompt, input_variables=["context"])
    
#     chain = load_qa_chain(summarizer, chain_type="stuff", prompt=prompt)
    
#     # Wrap the text in a Document object with page_content
#     input_data = {"input_documents": [Document(page_content=text)]}
    
#     summary = chain(input_data, return_only_outputs=True)
#     return summary["output_text"]

# # Function to process user input and return a response
# def process_user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(user_question)
    
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
#     return response["output_text"]

# # Main Streamlit app function
# def main():
#     st.set_page_config(page_title="Chat with PDF")
#     st.header("Chat with PDF using Gemini💁")

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
#         if st.button("Submit & Process", key="process_button"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Processing complete! Ask your questions below.")
#             else:
#                 st.warning("Please upload at least one PDF file.")
    
#     if 'pdf_texts' not in st.session_state:
#         st.session_state['pdf_texts'] = {}

#     if 'summary' not in st.session_state:
#         st.session_state['summary'] = ""

#     pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    
#     # Summarization input
#     pdfs_to_summarize = st.text_input("Enter the names of the PDFs you want to summarize, separated by commas")
    
#     if st.button("Summarize"):
#         if pdf_docs:
#         # Process user input for PDF names
#             pdf_names_input = [name.strip() for name in pdfs_to_summarize.split(",")]

#         # Initialize a variable to store combined text
#             text = ""

#         # Loop through all selected PDF names
#             for pdf_name, pdf_file in zip(pdf_names, pdf_docs):
#                 if pdf_name in pdf_names_input:
#                 # Only process the PDF if it's not already summarized
#                     if pdf_name not in st.session_state['pdf_texts']:
#                         st.session_state['pdf_texts'][pdf_name] = get_pdf_text([pdf_file])  # Store the PDF text in session

#                 # Combine the text from the selected PDFs
#                     text += st.session_state['pdf_texts'][pdf_name]
        
#         # Generate and store summaries for selected PDFs
#             for pdf_name in pdf_names_input:
#                 if pdf_name in st.session_state['pdf_texts']:
#                 # If the summary for this PDF has not been created yet
#                     if pdf_name not in st.session_state:
#                         summary = summarize_text(st.session_state['pdf_texts'][pdf_name])  # Create summary
#                         st.session_state[pdf_name] = summary  # Store the summary in session state
#                         st.write(f"Summary for {pdf_name}:", summary)  # Display the summary for this PDF
#                 else:
#                     st.error(f"Error: PDF '{pdf_name}' not found.")
                
#         else:
#             st.warning("Please upload PDF files before summarizing.")

        
#             # Generate summary only if there's combined text
#             if text:
#                 # If summary already exists for this PDF, skip summarization
#                 if pdf_names_input[0] not in st.session_state:
#                     # Save the summary for the current PDF
#                     st.session_state[pdf_names_input[0]] = summarize_text(text)  # Store summary by PDF name
#                     st.write("Summary:", st.session_state[pdf_names_input[0]])  # Display only this summary
#                 else:
#                     # If summary is already in session, display it
#                     st.write("Summary:", st.session_state[pdf_names_input[0]])

#             else:
#                 st.warning("Please upload PDF files before summarizing.")




#     # Display the summary if it exists
#     if st.session_state['summary']:
#         st.write("Summary:", st.session_state['summary'])

#     # Question input
#     user_question = st.text_input("Ask a Question from the PDF Files")
#     if user_question:
#         with st.spinner("Generating response..."):
#             response = process_user_input(user_question)
#             st.write("Reply: ", response)

# if __name__ == "__main__":
#     main()


import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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

# with right:
#     st.subheader("Session Snapshot")
#     st.write("Uploaded files:")
#     for f in (st.session_state.uploaded_files or []):
#         st.write(f"- {f.name}")
#     st.write("Stored extracted texts:")
#     for k, v in st.session_state.pdf_texts.items():
#         st.write(f"- {k} (chars: {len(v)})")
#     st.write("Summaries cached:")
#     for k in st.session_state.summaries.keys():
#         st.write(f"- {k}")
#     st.write(f"FAISS built: {st.session_state.faiss_built}")
#     st.write("---")
#     st.info("Embeddings: local sentence-transformers (no Google embedding quota required).")
#     st.info("LLM: Gemini (requires GOOGLE_API_KEY).")

# End of file
