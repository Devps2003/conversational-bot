from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # Configure Google Generative AI
# genai.configure(api_key=GOOGLE_API_KEY)

# # Create an instance of Pinecone
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])


# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_match(refined_query):
    index_name = "botttt"
    index = pc.Index(index_name)
    
    query_embedding = model.encode([refined_query]).tolist()[0]
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)


    if not results['matches']:
        return "No relevant context found."

    combined_texts = " ".join([match['metadata']['text'] for match in results['matches']])
    return combined_texts

llm = genai.GenerativeModel('gemini-1.5-flash')
def query_refiner(conversation, query):
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG:\n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    response = llm.generate_content(prompt)
    refined_query = response.text
    return refined_query


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string
