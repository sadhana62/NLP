import httpx
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("pageecho.ollama")

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5:1.5b"):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model

    async def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not reachable: {e}")
            return False

    async def list_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
        return [self.default_model]

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate completion response from Ollama model."""
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_format:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", "").strip()
                else:
                    logger.error(f"Ollama API error ({res.status_code}): {res.text}")
                    return f"[Error: Ollama returned status {res.status_code}]"
        except httpx.TimeoutException:
            logger.error("Ollama request timed out.")
            return "[Error: Ollama generation timed out. Try asking a shorter question or checking Ollama status.]"
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"[Error connecting to local Ollama server: {str(e)}]"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """Chat completion endpoint for conversational turn-taking."""
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    msg = data.get("message", {})
                    return msg.get("content", "").strip()
                else:
                    logger.error(f"Ollama chat error ({res.status_code}): {res.text}")
                    return f"[Error: Ollama chat returned status {res.status_code}]"
        except Exception as e:
            logger.error(f"Error calling Ollama chat: {e}")
            return f"[Error chatting with Ollama: {str(e)}]"
