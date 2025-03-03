from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
import chromadb
import requests
import json
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Inicializace FastAPI aplikace
app = FastAPI()

# CORS konfigurace
origins = [
    "http://dotazy.wz.cz",  # Povolte pouze konkrétní doménu, ne "*" pro bezpečnost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Povolené domény
    allow_credentials=True,
    allow_methods=["*"],  # Povolit všechny HTTP metody (POST, GET, OPTIONS)
    allow_headers=["*"],  # Povolit všechny hlavičky
)

# Načtení OpenAI API klíče z prostředí
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Chybí API klíč OpenAI. Nastavte proměnnou prostředí OPENAI_API_KEY.")

# Cesta k ChromaDB databázi
DB_PATH = "chroma_db/chroma.sqlite3"

# Inicializace ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# Načtení embeddingů z GitHubu
GITHUB_EMBEDDINGS_URL = "https://raw.githubusercontent.com/uzivatel/repozitar/main/embeddings.json"
response = requests.get(GITHUB_EMBEDDINGS_URL)
if response.status_code == 200:
    embeddings_data = response.json()
    for doc in embeddings_data:
        collection.add(documents=[doc["text"]], embeddings=[doc["embedding"]], ids=[doc["id"]])
else:
    raise RuntimeError("Nepodařilo se načíst embeddingy z GitHubu.")

# Kořenový endpoint (pro testování připojení)
@app.get("/")
def root():
    return {"message": "API je online! Použijte endpoint /ask pro odeslání dotazu."}

# Endpoint pro favicon.ico
@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# Model pro požadavky na API
class QueryRequest(BaseModel):
    query: str

# Endpoint pro zpracování dotazů na /ask
@app.post("/ask")
async def ask(request: QueryRequest):
    query = request.query
    query_embedding = [0.0] * 1536  # Zde by měl být embedding dotazu, zatím placeholder

    try:
        # Hledání nejpodobnějších dokumentů
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        if results["documents"]:
            relevant_docs = "\n".join(results["documents"][0])
        else:
            return {"answer": "Odpověď nelze najít v dostupných dokumentech."}
        
        # Použití OpenAI API s omezením na nalezené dokumenty
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Odpovídej pouze na základě poskytnutých dokumentů."},
                {"role": "user", "content": f"Na základě těchto dokumentů odpověz na dotaz:\n{relevant_docs}\n\nDotaz: {query}"}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    
    except openai.OpenAIError as e:
        print(f"Chyba při volání OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail="Chyba při komunikaci s OpenAI API.")
    
    except Exception as e:
        print(f"Neočekávaná chyba: {str(e)}")
        raise HTTPException(status_code=500, detail="Interní chyba serveru.")

# Spuštění aplikace
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Použije port, který je přidělen Renderem
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
