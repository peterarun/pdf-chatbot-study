# Set OpenAI API key from secrets
import hashlib
import logging
import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(page_title="Upload and Chat with PDF", page_icon="📜")
st.title("Upload and Chat with your own PDF 📜")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# Function to extract text from PDF
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Prepare and get the conversation langchain object
def get_conversation_chain(vectorstore_in: FAISS) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(temperature=0.7, model='gpt-4o-mini')

    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore_in.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

# Processing a single doc
def process_single_doc(pdf_doc: UploadedFile, vectorstore_in: FAISS):
    raw_text = get_pdf_text(pdf_doc)
    text = get_text_chunks(raw_text)
    vectorstore_in.add_texts(text)
    logging.info(f"Added file {pdf_doc} in vector store")

# Delegating PDF processing
def process_docs(pdf_docs: list[UploadedFile], vectorstore_in: FAISS):
    try:
        for pdf in pdf_docs:
            process_single_doc(pdf, vectorstore_in)

        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Get unique file hash
def get_file_hash(file: UploadedFile):
    contents = file.read()
    file.seek(0)  # Reset pointer so file can be read again later
    return hashlib.md5(contents).hexdigest()

# Process new uploads alone
def handle_new_uploads(current_files: list[UploadedFile], vectorstore_in: FAISS):

    new_file_found: bool = False
    for file in current_files:
        file_hash = get_file_hash(file)
        if file_hash not in st.session_state.processed_file_hashes:
            new_file_found = True
            st.session_state.processed_file_hashes.add(file_hash)

            # Process the file here as needed
            process_single_doc(file, vectorstore_in)
            st.write(f"📂 New file uploaded: {file.name}")

    if not new_file_found:
        st.write("📂 No New files found")

    return True

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        key="multi_pdf_uploader",
        accept_multiple_files=True
    )

    # Create embeddings
    embeddings = OpenAIEmbeddings()


    if "vectorstore" not in st.session_state:
        vectorstore: FAISS = FAISS.from_texts(["dummy"], embeddings)
        st.session_state.vectorstore = vectorstore
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        vectorstore: FAISS = st.session_state.vectorstore

    if st.button("Process") and pdf_docs:
        st.session_state.processComplete = False
        with st.spinner("Processing your PDFs..."):

            if "processed_file_hashes" not in st.session_state:
                st.session_state.processed_file_hashes = set()

            # success = process_docs(pdf_docs, vectorstore)
            success = handle_new_uploads(pdf_docs, vectorstore)
            if success:
                # Create conversation chain
                convChain: ConversationalRetrievalChain = get_conversation_chain(vectorstore)
                st.session_state.conversation = convChain
                st.session_state.processComplete = True

                st.success("Processing complete!")

# Main chat interface
if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about your documents:")

    if user_question:
        try:
            with st.spinner("Thinking..."):
                convChain: ConversationalRetrievalChain = st.session_state.conversation
                response = convChain({
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Display initial instructions
else:
    st.write("👈 Upload your PDFs in the sidebar to get started!")
