import os
import openai
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Nastavení pro logování pro debugování
logging.basicConfig(level=logging.DEBUG)

# Nastavení OpenAI API klíče
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializace FastAPI aplikace
app = FastAPI()

# Připojení k ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="documents")

# Kořenový endpoint, který vrátí zprávu o stavu API
@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Třída pro příjem dotazu
class QueryRequest(BaseModel):
    query: str

# Endpoint pro odesílání dotazu
@app.post("/ask")
def ask_question(request: QueryRequest):
    # Vytvoření embeddingu pro dotaz
    query_embedding = openai.Embedding.create(input=request.query, model="text-embedding-ada-002")["data"][0]["embedding"]

    # Dotaz na ChromaDB pro relevantní dokumenty
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    # Pokud jsou nalezeny dokumenty, odpověď z ChatGPT na základě těchto dokumentů
    if results['documents']:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Odpovídej pouze na základě poskytnutých dokumentů."},
                {"role": "user", "content": results['documents'][0]}
            ]
        )
        return {"answer": response["choices"][0]["message"]["content"]}
    else:
        return {"answer": "Na tento dotaz nemám odpověď ve své databázi."}

# Dynamické získání portu pro nasazení na Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

