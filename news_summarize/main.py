import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Define the summarization prompt
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# Create summarization chain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Function to extract news content from a URL
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Failed to fetch news from {url}: {e}"

# Streamlit UI
st.title("📰 News Article Summarizer")
st.markdown("Enter a URL to a news article and get a quick summary using Google's Gemini LLM.")

url = st.text_input("Enter the URL of the news article:")

if st.button("Summarize"):
    if url:
        with st.spinner("Fetching and summarizing the article..."):
            article = extract_news(url)
            if article.startswith("Failed to fetch"):
                st.error(article)
            else:
                summary = summarize_chain.run(article=article)
                st.success("Here is the summary:")
                st.text_area("Summary", summary, height=200)
    else:
        st.warning("Please enter a valid URL.")
