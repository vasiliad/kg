"""
Suite of standardized test tasks designed to assess:
1. Instruction adherence
2. Complex multi-step reasoning
3. Code generation & structured JSON extraction
"""
from typing import List, Dict, Any

VERIFICATION_TASKS: List[Dict[str, Any]] = [
    {
        "name": "reasoning_logic",
        "description": "Tests multi-step logical deduction",
        "prompt": (
            "A box contains 3 red balls, 4 green balls, and 5 blue balls. "
            "You pick two balls without replacement. "
            "What is the exact probability that both balls are of different colors? "
            "Show your step-by-step arithmetic and give the final reduced fraction."
        ),
        "expected_keywords": ["fraction", "probability", "33/66", "19/22", "different colors"]
    },
    {
        "name": "code_architecture",
        "description": "Tests clean Python architecture and error handling",
        "prompt": (
            "Write a production-grade Python function using asyncio that performs healthchecks "
            "on a list of HTTP URLs concurrently with a concurrency limit of 5 and a timeout of 3 seconds. "
            "Return a dictionary mapping URL to a boolean success flag."
        ),
        "expected_keywords": ["asyncio", "Semaphore", "ClientSession", "try", "except"]
    },
    {
        "name": "json_structured_output",
        "description": "Tests strict adherence to JSON format without Markdown commentary",
        "prompt": (
            "Extract entities from this sentence into valid JSON only (no markdown, no backticks):\n"
            "'NVIDIA Corporation released Nemotron-4 340B on June 2024 with 4096 context tokens.'\n"
            "Format: {\"company\": str, \"model\": str, \"release_date\": str, \"context_tokens\": int}"
        ),
        "expected_keywords": ["company", "model", "release_date", "context_tokens"]
    }
]
