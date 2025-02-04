import os
import openai
import chromadb
from docx import Document
import tiktoken
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio

# Použití proměnné prostředí pro OpenAI API klíč
openai.api_key = os.getenv("OPENAI_API_KEY")  # Načítáme API klíč z prostředí

client = chromadb.Client()
collection_name = "dokumenty_kolekce"
collection = client.get_or_create_collection(name=collection_name)

app = Flask(__name__)
CORS(app)  # Povolení CORS pro komunikaci s frontendem

# Funkce pro načítání dokumentů
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            doc_path = os.path.join(directory_path, filename)
            document = Document(doc_path)
            content = "\n".join([para.text for para in document.paragraphs if para.text.strip() != ""])
            documents.append((filename, content))
    return documents

# Funkce pro dělení textu
def split_text(text, max_tokens=1000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

# Funkce pro vytvoření embeddingů a jejich uložení do ChromaDB
async def create_embeddings(documents):
    for doc_name, content in documents:
        text_chunks = split_text(content, max_tokens=1000)
        embeddings = []
        ids = []
        metadatas = []
        documents_list = []

        for i, chunk in enumerate(text_chunks):
            try:
                response = await asyncio.wait_for(
                    openai.embeddings.create(input=chunk, model="text-embedding-ada-002"),
                    timeout=30  # Timeout pro OpenAI request
                )
                embedding = response.data[0].embedding
                ids.append(f"{doc_name}_{i}")
                embeddings.append(embedding)
                metadatas.append({"source": doc_name})
                documents_list.append(chunk)
            except asyncio.TimeoutError:
                print(f"Timeout při generování embeddingu pro chunk {i} souboru {doc_name}")
                continue

        # Přidání do ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_list
        )

# Funkce pro dotazování do ChromaDB
def query_chromadb(query, n_results=5):
    response = openai.embeddings.create(input=query, model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents"])
    return results.get("documents", [])

# API cesta pro chat
@app.route("/api/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Dotaz nesmí být prázdný"}), 400

    documents = load_documents_from_directory("./documents")
    if len(collection.get()["documents"]) == 0:
        asyncio.run(create_embeddings(documents))
    
    context_documents = query_chromadb(query)
    
    if context_documents:
        answer = generate_answer_with_assistant(query, context_documents)
    else:
        answer = "Bohužel, odpověď ve své databázi nemám."
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    from gunicorn.app.base import BaseApplication

    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)

        def load(self):
            return self.application

    options = {
        "bind": "0.0.0.0:" + os.getenv("PORT", "10000"),
        "workers": 2,
        "timeout": 120,
        "worker_class": "gevent"
    }
    GunicornApp(app, options).run()
