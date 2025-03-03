from fastapi import FastAPI, HTTPException
import chromadb
import os
from pydantic import BaseModel

# Nastavení cesty k databázi
DB_PATH = "chroma_db/chroma.sqlite3"

# Kontrola a smazání souboru, pokud existuje
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# Inicializace ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# FastAPI aplikace
app = FastAPI()

# Model dotazu
class QueryRequest(BaseModel):
    query_embedding: list[float]
    top_k: int = 3

@app.post("/query")
def query_chromadb(request: QueryRequest):
    try:
        results = collection.query(
            query_embeddings=[request.query_embedding],
            n_results=request.top_k
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "ChromaDB FastAPI server is running"}
