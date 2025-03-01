from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai
from fastapi.responses import FileResponse

# Inicializace FastAPI aplikace
app = FastAPI()

# Nastavení CORS
origins = [
    "http://dotazy.wz.cz",  # Povolit pouze tuto doménu
    "*",  # Nebo použít "*" pro povolení všech domén, ale to není doporučené pro produkci
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Seznam povolených domén
    allow_credentials=True,
    allow_methods=["*"],  # Povolit všechny HTTP metody (POST, GET, DELETE, atd.)
    allow_headers=["*"],  # Povolit všechny hlavičky
)

# Nastavení OpenAI API klíče
openai.api_key = os.getenv("OPENAI_API_KEY")

# Kořenový endpoint (pro testování připojení)
@app.get("/")
def root():
    return {"message": "API je online! Použijte endpoint /ask pro odeslání dotazu."}

# Endpoint pro favicon.ico (pokud máte soubor v adresáři "static")
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
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=query,
            max_tokens=100
        )
        answer = response.choices[0].text.strip()
        return {"answer": answer}
    except Exception as e:
        print(f"Chyba při zpracování dotazu: {str(e)}")  # Přidání logování pro chybové hlášky
        raise HTTPException(status_code=500, detail="Chyba při zpracování dotazu: " + str(e))

# Nastavení správného portu pro Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Použije port, který je přidělen Renderem
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
