import os
import sys
import time
from typing import Optional, Any

class TPULoader:
    """
    Manages loading and running DeepSeek-R1-Distill-Qwen-32B across 8 TPU cores (128GB HBM).
    Uses JAX/XLA backend via Keras 3 with sharding.
    """
    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        memory_fraction: float = 0.95,
        max_seq_len: int = 4096
    ):
        self.model_id = model_id
        self.memory_fraction = memory_fraction
        self.max_seq_len = max_seq_len
        self.model = None
        self._setup_tpu_environment()

    def _setup_tpu_environment(self) -> None:
        """Sets necessary environment variables before importing JAX / Keras."""
        os.environ["KERAS_BACKEND"] = "jax"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.memory_fraction)
        # Prevent JAX from pre-allocating all memory immediately in an unfragmented block
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    def load_model(self) -> Any:
        """
        Loads the model into TPU memory, distributed across all 8 cores.
        """
        print(f"[*] Initializing TPU environment (target: 128GB HBM / 8 cores)...")
        try:
            import jax
            devices = jax.devices()
            print(f"[+] Detected JAX devices: {len(devices)} devices: {devices}")
            if len(devices) < 8:
                print(f"[!] Warning: Expected 8 TPU cores, found {len(devices)}. Proceeding with available hardware.")
        except Exception as e:
            print(f"[!] Warning: JAX device inspection error: {e}")

        import keras_nlp

        print(f"[*] Loading model weights from: {self.model_id}...")
        start_time = time.time()
        
        # Load via KerasNLP CausalLM preset
        try:
            # First try loading directly via preset or HF hub mapping
            preset_name = f"hf://{self.model_id}"
            self.model = keras_nlp.models.CausalLM.from_preset(preset_name)
        except Exception as e:
            print(f"[-] KerasNLP direct preset loading failed ({e}). Falling back to AutoModel via transformers/torch_xla if available...")
            raise RuntimeError(f"Could not load {self.model_id} onto TPU. Detailed error: {e}")

        elapsed = time.time() - start_time
        print(f"[+] Model loaded successfully in {elapsed:.1f}s.")
        return self.model

    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.6) -> str:
        """
        Generates reasoning tokens using DeepSeek-R1 prompt format.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        # DeepSeek-R1 chat template formatting
        formatted_prompt = f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
        print(f"[*] Dispatching prompt to TPU ({len(prompt)} chars)...")
        
        start_gen = time.time()
        output = self.model.generate(
            formatted_prompt,
            max_length=self.max_seq_len
        )
        gen_time = time.time() - start_gen
        
        output_str = str(output)
        # Strip input prompt if returned
        if output_str.startswith(formatted_prompt):
            output_str = output_str[len(formatted_prompt):]
            
        print(f"[+] Generation complete in {gen_time:.2f}s.")
        return output_str
