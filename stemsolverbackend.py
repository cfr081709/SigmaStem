from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os

app = FastAPI()

# Allow frontend (localhost:5500) to access backend (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],  # your frontend origin here
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods including OPTIONS
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set your key in env variable for security

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set!")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    if "model" not in data or "messages" not in data:
        raise HTTPException(status_code=400, detail="Invalid request payload")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Request error: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"OpenAI API error: {exc.response.text}")
