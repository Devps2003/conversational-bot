from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import asyncio
import pinecone

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks to process.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunk_embeddings = embeddings.embed_documents(text_chunks)

    # Create an instance of Pinecone
    pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    # Create an index
    index_name = "botttt"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Set the dimension according to your SentenceTransformer model
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Connect to the index
    index = pc.Index(index_name)

    # Upsert the embeddings into the Pinecone index
    vectors = [
        {"id": f"text_chunk_{i}", "values": embedding, "metadata": {"text": chunk}}
        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings))
    ]
    index.upsert(vectors)

    return index

st.subheader("My name is FOS, how can I help you?")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Ensure an event loop is created
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Use a different model that might be compatible
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say use your knowledge to answer the question""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Container for chat history
response_container = st.container()
# Container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

with st.sidebar:
    st.title("Data")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process", key="process_pdfs"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text:
                st.error("No text extracted from the PDF files.")
            else:
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.error("Failed to split text into chunks.")
                else:
                    get_vector_store(text_chunks)
                    st.success("Done")
