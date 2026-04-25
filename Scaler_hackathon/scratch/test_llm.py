import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load env from ROOT
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
)

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10
    )
    print(f"Success: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
