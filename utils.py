from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key="4db3677a-d6bd-4829-aae8-9fc15b2a58cf")

# Check if the index exists and create it if it does not
index_name = "langchain-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Set the dimension according to your SentenceTransformer model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Changed to a supported region
    )

# Connect to the index
index = pc.Index(index_name)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, include_metadata=True)

    if not result['matches']:
        # If there are no matches, return an empty string or a default message
        return "No relevant context found."

    if len(result['matches']) == 1:
        # If there's only one match, return its text
        return result['matches'][0]['metadata']['text']
    else:
        # If there are two or more matches, return the text of the first two
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

# Use a different model that might be compatible
llm = genai.GenerativeModel('gemini-1.5-flash')

def query_refiner(conversation, query):
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\\n\\nCONVERSATION LOG: \\n{conversation}\\n\\nQuery: {query}\\n\\nRefined Query:"
    response = llm.generate_content(
        prompt
    )
    # Access the response correctly
    refined_query = response.text # or response.generated_text if result is not correct
    return refined_query

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string
