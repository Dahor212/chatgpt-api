from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
import chromadb
import requests
import sqlite3
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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Načtení OpenAI API klíče z prostředí
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Chybí API klíč OpenAI. Nastavte proměnnou prostředí OPENAI_API_KEY.")

client = openai.OpenAI(api_key=openai_api_key)

# Stažení databáze embeddingů z GitHubu
DB_URL = "https://github.com/Dahor212/chatgpt-api/raw/main/chroma_db/chroma.sqlite3"
DB_PATH = "chroma.sqlite3"

if not os.path.exists(DB_PATH):
    response = requests.get(DB_URL)
    if response.status_code == 200:
        with open(DB_PATH, "wb") as file:
            file.write(response.content)
    else:
        raise RuntimeError("Nepodařilo se stáhnout databázi embeddingů.")

# Připojení k ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection("documents")

# Model pro požadavky na API
class QueryRequest(BaseModel):
    query: str

# Kořenový endpoint
@app.get("/")
def root():
    return {"message": "API je online! Použijte endpoint /ask pro odeslání dotazu."}

# Endpoint pro favicon.ico
@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# Endpoint pro zpracování dotazů na /ask
@app.post("/ask")
async def ask(request: QueryRequest):
    query = request.query
    
    try:
        # Vyhledání podobných embeddingů v ChromaDB
        results = collection.query(query_texts=[query], n_results=3)
        documents = results["documents"] if results["documents"] else []
        
        if not documents:
            return {"answer": "Omlouvám se, ale tuto informaci v databázi nemám."}
        
        context = "\n".join(documents[0])
        
        # Volání OpenAI API s nalezeným kontextem
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Odpovídej pouze na základě poskytnutého kontextu."},
                {"role": "user", "content": f"Kontekst:\n{context}\n\nOtázka: {query}"}
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
    port = int(os.getenv("PORT", 8000))  # Výchozí port Render.com
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
