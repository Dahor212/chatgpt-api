import os
import requests
import chromadb
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from docx import Document  # Knihovna pro načítání .docx dokumentů

# Inicializace ChromaDB
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection_name = "dokumenty_kolekce"
collection = client_chroma.get_or_create_collection(name=collection_name)

app = Flask(__name__)
CORS(app)

# Endpoint pro sledování využití paměti
@app.route("/api/memory", methods=["GET"])
def memory_usage():
    mem = psutil.virtual_memory()
    return jsonify({
        "total": mem.total / 1024**2,
        "used": mem.used / 1024**2,
        "available": mem.available / 1024**2,
        "percent": mem.percent
    })

# Načtení embeddingů a obsahu dokumentů z GitHubu
def load_embeddings_from_github():
    url = "https://raw.githubusercontent.com/Dahor212/chatgpt-api/main/Embeddingy/embeddings.json"
    response = requests.get(url)
    if response.status_code == 200:
        embeddings_data = response.json()

        if isinstance(embeddings_data, dict) and "document" in embeddings_data and "embeddings" in embeddings_data:
            document_name = embeddings_data["document"]
            document_embeddings = embeddings_data["embeddings"]
            return {document_name: document_embeddings}

        print(f"Načtené embeddingy (ukázka): {json.dumps(embeddings_data)[:500]}")
        return embeddings_data
    else:
        raise Exception(f"Chyba při načítání embeddingů: {response.status_code}")

# Načtení obsahu dokumentu .docx
def load_document_content(doc_name):
    doc_path = f"./dokumenty/{doc_name}"
    try:
        doc = Document(doc_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Chyba při načítání dokumentu {doc_name}: {e}")
        return ""

# Výpočet kosinové podobnosti mezi dvěma vektory
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

# Vyhledání relevantních dokumentů
def query_chromadb(query, n_results=5):
    embeddings_data = load_embeddings_from_github()
    
    query_embedding = [0] * len(next(iter(embeddings_data.values())))  # Dummy embedding pro ilustraci
    
    results = []
    for doc_name, doc_embeddings in embeddings_data.items():
        if isinstance(doc_embeddings, list) and all(isinstance(e, list) for e in doc_embeddings):
            for embedding in doc_embeddings:
                similarity = cosine_similarity(query_embedding, embedding)
                results.append({
                    "document": doc_name,
                    "similarity": similarity
                })

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:n_results]

# Generování odpovědi s využitím předpřipraveného kontextu
def generate_answer_with_context(query, context_documents):
    if not context_documents:
        return "Bohužel, odpověď ve své databázi nemám."

    # Získání obsahu dokumentu
    document_name = context_documents[0]['document']
    document_content = load_document_content(document_name)
    
    print(f"Použitý kontext pro dotaz: {document_content}")

    return document_content  # Vrátí text dokumentu

@app.route("/", methods=["GET"])
def home():
    return "Aplikace běží!"

@app.route("/api/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Dotaz nesmí být prázdný"}), 400

    context_documents = query_chromadb(query)
    answer = generate_answer_with_context(query, context_documents)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True, workers=2)
