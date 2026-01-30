from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = splitter.create_documents(texts, metadatas=metadatas)

    vectordb = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory="chroma_store"
    )
    return vectordb


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    if len(question.split()) <= 2:
        question = "Explain " + question

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a question-answering system that must rely ONLY on the given context.\n"
                        "Do not add information that is not directly stated in the context.\n"
                        "Do not explain beyond what is written.\n"
                        "If multiple pieces of context are provided, merge them carefully.\n"
                        "If the answer is not explicitly found, say exactly: "
                        "'The answer is not available on the provided website.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            temperature=0,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("LLM ERROR:", e)
        return "LLM request failed."
