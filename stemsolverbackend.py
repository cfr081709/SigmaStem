import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()

# CORS - allow frontend calls from localhost:5500
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Use environment variable or fallback
HF_API_KEY = 
HF_MODEL = "gpt2"  # Use a known public model for testing

if not HF_API_KEY:
    raise RuntimeError("Hugging Face API key is missing!")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    logging.info(f"Received data: {data}")

    if "messages" not in data:
        raise HTTPException(status_code=400, detail="Missing 'messages' in request")

    # Build prompt from messages
    prompt = ""
    for msg in data["messages"]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        prompt += f"{role.capitalize()}: {content}\n"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    logging.info(f"Sending request to {url}")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            # Use synchronous .json(), not await
            result = response.json()
            logging.info(f"Model response: {result}")

            generated_text = (
                result[0].get("generated_text", "")
                if isinstance(result, list) and result
                else str(result)
            )

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        }
                    }
                ]
            }

        except httpx.RequestError as exc:
            logging.error(f"Request error: {exc}")
            raise HTTPException(status_code=500, detail=f"Request error: {exc}")
        except httpx.HTTPStatusError as exc:
            logging.error(f"Hugging Face API error: {exc.response.text}")
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Hugging Face API error: {exc.response.text}",
            )
