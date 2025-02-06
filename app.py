import os
import requests
import chromadb
import psutil
import openai
import numpy as np
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

# Načtení embeddingů dokumentů z GitHubu
def load_embeddings_from_github():
    url = "https://raw.githubusercontent.com/Dahor212/chatgpt-api/main/Embeddingy/embeddings.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        embeddings_data = response.json()
        if not isinstance(embeddings_data, dict):
            print("⚠️ Chyba: Načtený soubor nemá správný formát JSON.")
            return {}
        return embeddings_data
    except requests.RequestException as e:
        print(f"⚠️ Chyba při načítání embeddingů: {e}")
        return {}
    except json.JSONDecodeError:
        print("⚠️ Chyba: Embeddingy nelze dekódovat jako JSON.")
        return {}

# Výpočet kosinové podobnosti mezi dvěma vektory
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)

    if vec1.shape != vec2.shape:
        return 0  # Pokud jsou různé délky, vrátíme 0
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

# Vyhledání relevantních dokumentů
def query_chromadb(query, n_results=5):
    embeddings_data = load_embeddings_from_github()
    
    if not embeddings_data:
        return []

    # Náhradní embedding dotazu (musí mít stejnou délku jako dokumenty)
    first_doc_embedding = next(iter(embeddings_data.values()))
    if not first_doc_embedding:
        print("⚠️ Chyba: Žádné embeddingy nejsou dostupné.")
        return []
    
    embedding_length = len(first_doc_embedding[0])
    query_embedding = np.zeros(embedding_length).tolist()  # Správné vytvoření nulového vektoru

    results = []
    for doc_name, doc_embeddings in embeddings_data.items():
        for embedding in doc_embeddings:
            similarity = cosine_similarity(query_embedding, embedding)
            results.append({
                "document": doc_name,
                "similarity": similarity
            })

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:n_results]

# Generování odpovědi pomocí GPT-4
def generate_answer_with_gpt(query, context_documents):
    if not context_documents:
        return "Bohužel, odpověď ve své databázi nemám."

    context_texts = "\n\n".join([doc['document'] for doc in context_documents])
    prompt = f"Na základě následujících dokumentů odpověz na dotaz:\n\n{context_texts}\n\nDotaz: {query}\nOdpověď:"

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Jsi AI asistent, který odpovídá na základě dokumentů."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"⚠️ Chyba při volání OpenAI API: {str(e)}")
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True)
