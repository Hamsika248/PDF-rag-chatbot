import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ─────────────────────────────────────────────
# 1. Load environment variables
# ─────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")

# ─────────────────────────────────────────────
# 2. Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Personal PDF Chatbot")
st.markdown(
    "Upload any PDF and ask natural-language questions about its content. "
    "Powered by **LangChain + FAISS + Gemini**."
)

# ─────────────────────────────────────────────
# 3. Helper functions
# ─────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def split_text_into_chunks(text: str) -> list[str]:
    """Split raw text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list[str]) -> FAISS:
    """Embed chunks and store them in a FAISS vector index."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def build_conversation_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """Build a conversational RAG chain backed by the FAISS retriever."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
    )
    return chain

# ─────────────────────────────────────────────
# 4. Session state initialisation
# ─────────────────────────────────────────────
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ─────────────────────────────────────────────
# 5. Sidebar — PDF upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        if st.button("🔄 Process PDF", use_container_width=True):
            with st.spinner("Reading and indexing your PDF…"):
                # Extract → chunk → embed → store
                raw_text = extract_text_from_pdf(uploaded_file)
                if not raw_text.strip():
                    st.error("Could not extract text from this PDF. Try a different file.")
                else:
                    chunks = split_text_into_chunks(raw_text)
                    vectorstore = build_vectorstore(chunks)
                    st.session_state.conversation = build_conversation_chain(vectorstore)
                    st.session_state.chat_history = []
                    st.session_state.pdf_processed = True
                    st.success(
                        f"✅ PDF processed! {len(chunks)} chunks indexed. Start chatting below."
                    )

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a PDF\n"
        "2. Click **Process PDF**\n"
        "3. Ask any question about it\n\n"
        "The chatbot retrieves the most relevant sections "
        "from your document and answers using Gemini."
    )

# ─────────────────────────────────────────────
# 6. Main chat area
# ─────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload and process a PDF from the sidebar to get started.")
else:
    # Render existing chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about your PDF…")
    if user_question:
        # Display user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get answer from RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = st.session_state.conversation({"question": user_question})
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
