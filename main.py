from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
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

    try:
        # Použití správného API volání dle nové OpenAI knihovny
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            api_key=openai_api_key  # Použití klíče explicitně
        )

        answer = response["choices"][0]["message"]["content"].strip()
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
