import os
import requests
import chromadb
import psutil
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from dotenv import load_dotenv

# Načtení API klíče z .env souboru
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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

# Načtení embeddingů z GitHubu
def load_embeddings_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Chyba při načítání embeddingů: {response.status_code}")

# Výpočet kosinové podobnosti mezi dvěma vektory
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

# Funkce pro vyhledání relevantních dokumentů
def query_chromadb(query, n_results=5):
    embeddings_data = load_embeddings_from_github(
        "https://raw.githubusercontent.com/Dahor212/chatgpt-api/main/Embeddingy/embeddings.json"
    )
    query_embeddings_data = load_embeddings_from_github(
        "https://raw.githubusercontent.com/Dahor212/chatgpt-api/main/Embeddingy/query_embeddings.json"
    )

    # Získání embeddingu dotazu (pokud existuje)
    query_embedding = query_embeddings_data.get(query)
    if query_embedding is None:
        print(f"Varování: Pro dotaz '{query}' nebyl nalezen embedding.")
        return []

    results = []
    for doc_name, doc_embeddings in embeddings_data.items():
        if isinstance(doc_embeddings, list) and all(isinstance(e, list) for e in doc_embeddings):
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

# Generování odpovědi pomocí GPT-4
def generate_answer_with_gpt(query, context_documents):
    if not context_documents:
        return "Bohužel, odpověď ve své databázi nemám."

    context_texts = "\n\n".join([doc['document'] for doc in context_documents])
    prompt = f"Na základě následujících dokumentů odpověz na dotaz:\n\n{context_texts}\n\nDotaz: {query}\nOdpověď:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Jsi AI asistent, který odpovídá na základě dokumentů."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Chyba při volání OpenAI API: {str(e)}")
        return "Došlo k chybě při generování odpovědi."

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
