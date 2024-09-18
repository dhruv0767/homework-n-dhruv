import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import cohere

def run():
    st.title("URL Content Summarizer and Q&A")

    # Load API keys from secrets.toml
    openai.api_key = st.secrets["openai"]["api_key"]
    anthropic_api_key = st.secrets["anthropic"]["api_key"]
    cohere_api_key = st.secrets["cohere"]["api_key"]

    # Initialize clients
    anthropic = Anthropic(api_key=anthropic_api_key)
    cohere_client = cohere.Client(cohere_api_key)

    # Sidebar options
    st.sidebar.header("Settings")
    
    url1 = st.sidebar.text_input("Enter the first URL:")
    url2 = st.sidebar.text_input("Enter the second URL (optional):")

    llm_vendor = st.sidebar.selectbox(
        "Select the LLM vendor:",
        ("OpenAI", "Anthropic", "Cohere")
    )

    llm_model = st.sidebar.selectbox(
        "Select the LLM model:",
        ("GPT-3.5" if llm_vendor == "OpenAI" else "GPT-4") if llm_vendor == "OpenAI" else
        ("Claude" if llm_vendor == "Anthropic" else "Claude-instant") if llm_vendor == "Anthropic" else
        ("Command" if llm_vendor == "Cohere" else "Command-Light")
    )

    # Main content
    user_question = st.text_input("Ask a question about the content:")

    def read_url_content(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find('article') or soup.find(id="bodyContent") or soup.find('main')
            return content_div.get_text(separator=' ', strip=True) if content_div else "Could not extract the main content of the page."
        except requests.RequestException as e:
            st.error(f"Error reading {url}: {e}")
            return None

    def get_llm_response(prompt):
        if llm_vendor == "OpenAI":
            model = "gpt-3.5-turbo" if llm_model == "GPT-3.5" else "gpt-4"
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            return (chunk['choices'][0]['delta'].get('content', '') for chunk in response)

        elif llm_vendor == "Anthropic":
            model = "claude-2" if llm_model == "Claude" else "claude-instant-1"
            response = anthropic.completions.create(
                model=model,
                prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                max_tokens_to_sample=1000,
                stream=True
            )
            return (chunk.completion for chunk in response)

        elif llm_vendor == "Cohere":
            model = "command" if llm_model == "Command" else "command-light"
            response = cohere_client.generate(
                model=model,
                prompt=prompt,
                max_tokens=1000,
                stream=True
            )
            return (chunk.text for chunk in response)

    if url1:
        content1 = read_url_content(url1)
        content2 = read_url_content(url2) if url2 else ""
        
        if content1:
            st.write("Content from URL 1 (First 500 characters):")
            st.write(content1[:500] + "...")
            
            if content2:
                st.write("Content from URL 2 (First 500 characters):")
                st.write(content2[:500] + "...")

            if user_question:
                prompt = f"Based on the following content:\n\nURL 1: {content1}\n\nURL 2: {content2}\n\nPlease answer this question: {user_question}"
                
                try:
                    st.subheader("Answer:")
                    answer_placeholder = st.empty()
                    full_answer = ""
                    
                    for token in get_llm_response(prompt):
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                    
                    answer_placeholder.markdown(full_answer)
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    run()