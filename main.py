import uvicorn
import logging
from fastapi import FastAPI
from app.routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO)

def create_app():
    app = FastAPI(title="LLM Service (Qwen2.5B + local)", version="1.0")
    app.include_router(chat_router, prefix="", tags=["chat"])
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8016, reload=True)