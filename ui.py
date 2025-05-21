import gradio as gr
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load the saved vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define RAG chain using Granite3.2:2b
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Answer the following question using the given context.\n\nContext:\n{formatted_context}\n\nQuestion:\n{question}"
    
    response = ollama.chat(
        model='granite3.2:2b',
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    return response['message']['content']

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs="text",
    outputs="text",
    title="Gym Dataset Q&A",
    description="Ask questions based on the gym workout CSV dataset."
)

iface.launch()
