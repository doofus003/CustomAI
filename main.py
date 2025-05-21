from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

# Load data
loader = CSVLoader(file_path="data/gym.csv", encoding="utf-8")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
splits = text_splitter.split_documents(docs)

# Create embeddings and persist vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"
)
vectorstore.persist()  # ✅ Save to disk

# Define LLM call
def ollama_llm(question, context):
    formatted_prompt = f"Answer the following question using the given context.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    response = ollama.chat(
        model='granite3.2:2b',
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    return response['message']['content']

# Setup RAG chain
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Optional: test it
result = rag_chain("What is Task Decomposition?")
print(result)
