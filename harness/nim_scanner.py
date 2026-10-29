import requests
import json
from typing import List, Dict, Any, Optional

class NIMScanner:
    """
    Scans and catalogues active models on NVIDIA NIM API (integrate.api.nvidia.com).
    Inspects model endpoints, supported modalities, and recommended sampling params.
    """
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def fetch_available_models(self) -> List[Dict[str, Any]]:
        """
        Queries the /models endpoint to retrieve the list of currently deployed NIM models.
        """
        url = f"{self.base_url}/models"
        print(f"[*] Querying NVIDIA NIM models catalog from: {url}...")
        from harness.nim_pacer import nim_pacer
        nim_pacer.wait_for_slot()
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 401:
                print("[-] Error: 401 Unauthorized. NVIDIA API key is invalid or rotated!")
                return []
            if response.status_code != 200:
                print(f"[-] Warning: Failed to fetch models. Status: {response.status_code}, Body: {response.text}")
                return []
            
            data = response.json()
            models_list = data.get("data", [])
            print(f"[+] Found {len(models_list)} available models on NVIDIA NIM.")
            return models_list
        except Exception as e:
            print(f"[-] Network exception while reaching NVIDIA NIM: {e}")
            return []

    def detect_catalog_diff(self, current_models: List[Dict[str, Any]], previous_state_path: str = "active_models.json") -> Dict[str, Any]:
        """
        Calculates diff between currently available NIM models and previously saved state:
        - newly_added: models deployed since last scan
        - removed: models deprecated/taken down (e.g. full glm-5.3)
        - flagship_newcomers: new models with high-capability architecture markers
        """
        current_ids = {m.get("id") for m in current_models if m.get("id")}
        previous_ids = set()
        
        try:
            with open(previous_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                active_fleet = state.get("active_fleet", [])
                previous_ids = {m.get("id") for m in active_fleet if isinstance(m, dict) and m.get("id")}
        except Exception:
            pass

        new_ids = current_ids - previous_ids if previous_ids else set()
        removed_ids = previous_ids - current_ids if previous_ids else set()

        flagship_keywords = [
            "ultra", "super", "nemotron", "550b", "340b", "253b", "120b", "70b",
            "reasoning", "r1", "v4", "v5", "glm-5", "minimax", "step-5", "kimi"
        ]
        
        flagship_newcomers = [
            m_id for m_id in new_ids 
            if any(kw in m_id.lower() for kw in flagship_keywords)
        ]

        if removed_ids:
            print(f"[!] CATALOG ALERT: {len(removed_ids)} models were removed/retired from NIM: {list(removed_ids)}")
        if new_ids:
            print(f"[+] CATALOG ALERT: {len(new_ids)} new models discovered on NIM!")
            if flagship_newcomers:
                print(f"[🔥] ULTRA FLAGSHIP NEWCOMERS DETECTED: {flagship_newcomers}")

        return {
            "new_count": len(new_ids),
            "removed_count": len(removed_ids),
            "new_ids": list(new_ids),
            "removed_ids": list(removed_ids),
            "flagship_newcomers": flagship_newcomers
        }

    def filter_high_tier_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters models strictly to the Ultra-Tier NIM fleet:
        Detects both known Ultra candidates and any newly arrived flagships (550B, Ultra, Super, Flash, Reasoning).
        """
        ultra_tier_keywords = [
            "minimax", "m3.1", "glm-5.3", "glm5.3", 
            "v4.1", "deepseek-v4.1", "550b", "step-5", "step5",
            "nemotron-3-ultra", "nemotron-3.5", "nemotron-4-340b", "kimi-k2.6"
        ]
        selected = []
        for m in models:
            m_id = m.get("id", "").lower()
            if any(kw in m_id for kw in ultra_tier_keywords):
                selected.append({
                    "id": m.get("id"),
                    "created": m.get("created"),
                    "owned_by": m.get("owned_by", "nvidia"),
                    "recommended_params": self._get_recommended_params(m.get("id"))
                })
        print(f"[+] Filtered {len(selected)} Ultra-tier candidates from catalog.")
        return selected

    def _get_recommended_params(self, model_id: str) -> Dict[str, Any]:
        """
        Provides developer-recommended hyperparameters for Ultra-Tier models.
        """
        model_lower = model_id.lower()
        if "minimax" in model_lower or "m3.1" in model_lower:
            # MiniMax M3.1 Flash: high-throughput reasoning
            return {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096}
        elif "glm-5.3" in model_lower or "glm5.3" in model_lower:
            # GLM-5.3 Flash: agentic coding & tool orchestration
            return {"temperature": 0.5, "top_p": 0.85, "max_tokens": 4096}
        elif "v4.1" in model_lower or "deepseek" in model_lower:
            # DeepSeek V4.1 Flash: fast multi-token reasoning
            return {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096}
        elif "550b" in model_lower or "ultra" in model_lower:
            # NVIDIA 550B Ultra: massive flagship architecture
            return {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096}
        elif "step-5" in model_lower or "step5" in model_lower:
            # Step-5 Preview: reasoning & multimodal preview
            return {"temperature": 0.6, "top_p": 0.9, "max_tokens": 4096}
        return {"temperature": 0.6, "top_p": 0.9, "max_tokens": 2048}
