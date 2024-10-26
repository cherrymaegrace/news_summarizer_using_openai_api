import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

def call_model(user_input):
    system_prompt = """
    
    You are an advanced language model designed to assist users in summarizing news articles effectively. Your primary goal is to help users condense complex information into clear and concise summaries while maintaining the core message and essential details.

    Instructions:

    Information Gathering: When users provide excerpts or main points from an article, analyze the content for key themes, facts, and arguments. If the information is insufficient for a comprehensive summary, politely request additional details.

    Summarization: Create a summary that:

    Highlights the main ideas and important facts.
    Maintains the original tone and context of the article.
    Is clear, concise, and free of jargon, making it accessible to a broad audience.
    User Engagement: Encourage users to provide specific sections or themes they want emphasized in the summary. Offer clarifications or ask follow-up questions if needed to ensure the summary meets their expectations.

    Examples: If appropriate, provide examples of well-structured summaries to guide users in how to present their excerpts or main points.

    Respect Copyright: Remind users to share only the text they are comfortable providing and to respect copyright when discussing articles.

    By following these guidelines, you will help users effectively understand and communicate the essence of news articles.

    """

    struct = [{"role": "system", "content": system_prompt}]

    struct.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages = struct
    )

    output = response.choices[0].message.content
    struct.append({"role": "assistant", "content": output})

    return output


def main():
    st.title("News Summarizer App")

    # Add API key input in the sidebar
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    if api_key and len(api_key) != 164:
        st.sidebar.error("Invalid API key. It should be 164 characters long.")

    # Add option menu in the sidebar
    with st.sidebar:
        st.header("Navigation")
        selected_section = option_menu("Go to", ["Home", "Summarizer", "About"])

    if selected_section == "Home":
        st.header("Welcome to News Summarizer App")
        st.write("Select 'Summarizer' from the sidebar to start summarizing news articles.")
        st.write("You can find more information about the app in the 'About' section.")

    elif selected_section == "Summarizer":
        st.header("Summarize News Article")
        user_input = st.text_area("Paste your news article here:", height=200)
        if st.button("Summarize") and api_key:
            openai.api_key = api_key
            output = call_model(user_input)
            
            # Create two columns for alignment
            col1, col2 = st.columns(2)
            
            # Display original article in the left column
            with col1:
                st.text_area("Original Article:", user_input, height=300, key="original_article")
            
            # Display summary in the right column
            with col2:
                st.text_area("Summary:", output, height=300, key="summary_output")
        elif not api_key:
            st.warning("Please enter a valid API key in the sidebar.")
        else:
            st.warning("Please enter a news article to summarize.")

        # Add model information below the summarizer
        st.subheader("Model Information")
        st.info("This summarizer uses the GPT-4 model from OpenAI.")

    elif selected_section == "About":
        st.header("About")
        st.info("This is a news summarizer app using Streamlit and OpenAI's GPT-4 model.")
        st.write("Created by: Your Name")
        st.write("Version: 1.0")
        st.write("Last Updated: 2023-04-15")

if __name__ == "__main__":
    main()
