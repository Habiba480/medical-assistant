import requests
from src.constants import LLM_API_URL, MODEL_NAME

def query_llm(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.5,
        "stream": False
    }

    try:
        response = requests.post(LLM_API_URL, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception:
        return "⚠️ Could not reach the language model. Check if LM Studio is running."
