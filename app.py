import os
import requests
import chromadb
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from sentence_transformers import SentenceTransformer  # Knihovna pro generování embeddingu pro dotaz
import openai  # Knihovna pro volání GPT modelu

# Inicializace ChromaDB
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection_name = "dokumenty_kolekce"
collection = client_chroma.get_or_create_collection(name=collection_name)

# Inicializace modelu pro generování embeddingu pro dotaz
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Můžete použít jiný model dle potřeby

# Inicializace OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ujistěte se, že máte nastavený API klíč pro OpenAI

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

# Načtení embeddingů z GitHubu
def load_embeddings_from_github():
    url = "https://raw.githubusercontent.com/Dahor212/chatgpt-api/main/Embeddingy/embeddings.json"
    response = requests.get(url)
    if response.status_code == 200:
        embeddings_data = response.json()
        return embeddings_data
    else:
        raise Exception(f"Chyba při načítání embeddingů: {response.status_code}")

# Výpočet kosinové podobnosti mezi dvěma vektory
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

# Vyhledání relevantních dokumentů
def query_chromadb(query, n_results=5):
    embeddings_data = load_embeddings_from_github()

    # Generování embeddingu pro dotaz
    query_embedding = embedder.encode([query])[0]  # Generování embeddingu pro dotaz

    results = []
    for doc_name, doc_embeddings in embeddings_data.items():
        if isinstance(doc_embeddings, list) and all(isinstance(e, list) for e in doc_embeddings):  # Ověření struktury
            for embedding in doc_embeddings:
                similarity = cosine_similarity(query_embedding, embedding)
                results.append({
                    "document": doc_name,
                    "similarity": similarity
                })
        else:
            print(f"Varování: Neočekávaný formát embeddingů pro {doc_name}")

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:n_results]

# Volání OpenAI API pro generování odpovědi na základě kontextu
def generate_answer_with_gpt(query, context_documents):
    if not context_documents:
        return "Bohužel, odpověď ve své databázi nemám."

    # Sestavení kontextu pro GPT
    context = "\n\n".join([doc['document'] for doc in context_documents])

    # Volání OpenAI API pro generování odpovědi na základě kontextu
    response = openai.Completion.create(
        engine="text-davinci-003",  # Nebo jiný model dle potřeby
        prompt=f"Na základě následujícího kontextu odpověz na tento dotaz:\n\nKontext:\n{context}\n\nDotaz: {query}\n\nOdpověď:",
        max_tokens=200,  # Počet tokenů pro odpověď
        temperature=0.7  # Nastavení pro kreativitu odpovědi
    )

    return response.choices[0].text.strip()

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
    answer = generate_answer_with_gpt(query, context_documents)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True, workers=2)
