import streamlit as st
from crawler import get_all_links, extract_text_and_title
from rag_pipeline import process_website, get_answer

st.set_page_config(page_title="Website RAG Chatbot")
st.title("Website RAG Chatbot (Groq Powered)")

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "messages" not in st.session_state:
    st.session_state.messages = []

url = st.text_input("Enter Website URL")

if st.button("Index Website") and url:
    st.session_state.messages = []
    with st.spinner("Processing website..."):
        links = get_all_links(url)

        texts, metadatas = [], []
        for link in links:
            text, title = extract_text_and_title(link)
            if len(text) > 100:
                texts.append(text)
                metadatas.append({"source": link, "title": title})

        if texts:
            st.session_state.vectordb = process_website(texts, metadatas)
            st.success("Website indexed successfully!")
        else:
            st.error("No readable content found.")


# ✅ HANDLE INPUT FIRST
if st.session_state.vectordb:
    user_input = st.chat_input("Ask a question")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            answer = get_answer(user_input, st.session_state.vectordb)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ✅ THEN DISPLAY MESSAGES
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
