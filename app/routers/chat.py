from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import List, Dict
import logging
import torch
import json
from app.services.model_qwen import qwen_model

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat")
async def chat_inference(payload: Dict):
    """
    Принимает JSON вида:
    {
      "query": "What is the cyber law?",
      "context": [
        {"doc": "cyberlaw.pdf", "chunk": "some relevant text..."},
        {"doc": "ai_intro.pdf", "chunk": "some other text..."}
      ]
    }
    """
    try:
        query = payload["query"]
        context_chunks = payload.get("context", [])

        # Формируем системный prompt
        # (или можно прописать "system" role, если хочешь)
        # контекст
        context_str = ""
        for c in context_chunks:
            context_str += f"Document: {c['doc']} => {c['chunk']}\n"

        # Итоговый prompt
        final_prompt = f"""
You are an intelligent assistant with domain knowledge from multiple documents.
Use the following context to answer the user's question.

Instructions:
- Return the result as a JSON object.
- The JSON must contain two fields:
  - "answer": your full detailed answer.
  - "documents": a list of document names you used in your answer.

Context:
{context_str}

User: {query}
Assistant:
""".strip()


        # Генерация (псевдострим)
        answer = qwen_model.generate(final_prompt)

        # Разбиваем answer на токены и стримим
        return {"answer": answer}


    except Exception as e:
        logger.error(f"❌ Error in chat_inference: {e}")
        raise HTTPException(500, "LLM generation failed")