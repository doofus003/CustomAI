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

# Load CSS from external file
with open("pretty.css", "r") as f:
    custom_css = f.read()

# Gradio interface with dark-themed input/output
with gr.Blocks(css=custom_css) as iface:
    with gr.Column():
        gr.Markdown("""
        <div class="header">
            <h1>🏋️ Gym Dataset Q&A</h1>
            <p>Ask questions about workout routines, exercises, and fitness tips</p>
        </div>
        """)
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Your fitness question",
                placeholder="What's the best exercise for building chest muscles?",
                lines=3,
                elem_classes=["response"]
            )
        
        submit_btn = gr.Button("Get Answer", variant="primary")
        
        output = gr.Textbox(
            label="Answer",
            lines=3,
            elem_classes=["response"],
            interactive=False
        )
        
        gr.Markdown("""
        <div class="footer">
            Thank you for using our fitness assistant! 💪<br>
            We hope this helps you with your workout journey.
        </div>
        """)
    
    submit_btn.click(
        fn=rag_chain,
        inputs=question_input,
        outputs=output
    )

iface.launch()
