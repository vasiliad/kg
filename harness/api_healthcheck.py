import requests
import time
from typing import Dict, Any, List, Optional
from harness.prompt_suite import VERIFICATION_TASKS

class APIHealthChecker:
    """
    Validates API key status, model availability, latency, and response quality across NVIDIA NIM endpoints.
    Catches 401 Unauthorized (key rotation) and 429 Rate Limits.
    """
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def check_key_validity(self) -> Dict[str, Any]:
        """
        Quick probe to confirm if the current NVIDIA NIM API key is alive.
        """
        test_url = f"{self.base_url}/models"
        try:
            res = requests.get(test_url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                return {"valid": True, "status_code": 200, "message": "Key is active and authorized."}
            elif res.status_code == 401:
                return {"valid": False, "status_code": 401, "message": "ROTATION ALERT: Key revoked or expired (401 Unauthorized)!"}
            elif res.status_code == 403:
                return {"valid": False, "status_code": 403, "message": "FORBIDDEN: Key lacks permissions for this endpoint (403 Forbidden)."}
            else:
                return {"valid": False, "status_code": res.status_code, "message": f"Unexpected response: {res.text}"}
        except Exception as e:
            return {"valid": False, "status_code": 0, "message": f"Connection error: {e}"}

    def test_model_inference(
        self,
        model_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a verification task against a specific NIM model.
        Measures latency, adherence, and completion status.
        """
        params = params or {"temperature": 0.6, "top_p": 0.9, "max_tokens": 512}
        task = VERIFICATION_TASKS[0] # Test logic / reasoning
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": task["prompt"]}
            ],
            "temperature": params.get("temperature", 0.6),
            "top_p": params.get("top_p", 0.9),
            "max_tokens": params.get("max_tokens", 512)
        }

        url = f"{self.base_url}/chat/completions"
        start_t = time.time()
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=30)
            latency = time.time() - start_t
            
            if res.status_code == 200:
                body = res.json()
                choice = body.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                
                # Check keyword presence
                hits = sum(1 for kw in task["expected_keywords"] if kw.lower() in content.lower())
                pass_score = (hits / len(task["expected_keywords"])) * 100.0

                return {
                    "model_id": model_id,
                    "status": "HEALTHY",
                    "latency_sec": round(latency, 2),
                    "test_task": task["name"],
                    "keyword_score": pass_score,
                    "sample_response": content[:200] + "..." if len(content) > 200 else content,
                    "recommended_params": params
                }
            else:
                return {
                    "model_id": model_id,
                    "status": f"FAILED_HTTP_{res.status_code}",
                    "latency_sec": round(latency, 2),
                    "error": res.text
                }
        except Exception as e:
            return {
                "model_id": model_id,
                "status": "NETWORK_ERROR",
                "error": str(e)
            }
