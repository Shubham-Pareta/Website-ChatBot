from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load FLAN-T5 model directly (no pipeline)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)
    return Chroma.from_documents(docs, embedding_model)


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = " ".join(d.page_content[:200] for d in docs)

    prompt = f"Answer the question using the context.\nContext: {context}\nQuestion: {question}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()
