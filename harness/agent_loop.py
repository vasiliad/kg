"""
ReAct / Tool-Use Loop for DeepSeek-R1-32B on Kaggle TPU.
Provides the system prompt and iterative execution cycle:
Thought -> Action: tool_name(args) -> Observation -> Next Thought -> Final Answer
"""
import re
import json
from typing import Dict, Any, List
from harness.tools import AgentToolbox

AGENT_SYSTEM_PROMPT = """You are the Lead Autonomous AI Systems Architect running locally on a 128GB TPU cluster.
Your mission is to discover, evaluate, and configure the best external models from NVIDIA NIM for our project.

You have access to the following live tools:
1. `web_search(query="...")`: Search the web for current AI benchmarks (LMSYS Arena, LiveBench), rankings, and announcements.
2. `fetch_webpage(url="...")`: Read official documentation on build.nvidia.com or developer sites to extract recommended sampling parameters and verified context limits.
3. `query_nim_catalog()`: Get the raw list of currently active model IDs on NVIDIA NIM API.
4. `probe_nim_context_window(model_id="...")`: Actively inspect the EXACT incoming/outgoing context window limit on NVIDIA NIM (NIM often sets limits smaller than base models, e.g. 32k or 132k or 256k tokens).
5. `test_nim_model(model_id="...", temperature=0.6, top_p=0.95, max_tokens=2048, system_prompt="...", test_prompt="...", reasoning_effort=100)`: Send a live inference test to check responsiveness, latency, and response quality.

CRITICAL PARAMETER RULES:
- Context Windows: Do not assume standard HuggingFace context sizes! On NVIDIA NIM, the context window is often customized or constrained by NVIDIA infrastructure (e.g. 32k, 128k, 132k, 256k). You MUST inspect NVIDIA NIM documentation and use `probe_nim_context_window(model_id=...)` to record the exact server-side input context limit.
- DeepSeek Reasoning Effort: For DeepSeek models (and reasoning flash variants), thinking effort is specified in PERCENTAGES from 0 to 100 (maximum = 100), NOT as low/medium/high text strings! Always configure `reasoning_effort` as an integer percentage (e.g., `reasoning_effort=100` for maximum depth).

To use a tool, format your output strictly as:
Action: tool_name(argument="value", ...)

When you receive the Observation, analyze it inside your <think> tags, and then take the next action or provide the final verdict.

When you have completed your investigation, provide your final report strictly in JSON format as:
```json
{
  "best_models": [
    {
      "model_id": "...",
      "role": "...",
      "rationale": "...",
      "nim_context_window": "132k (135168 tokens)",
      "recommended_parameters": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096
      },
      "system_prompt_strategy": "...",
      "test_result": "PASSED"
    }
  ]
}
```
Do not rely on hardcoded assumptions. Use your tools to verify live information.
"""

class ReActAgent:
    def __init__(self, tpu_loader, toolbox: AgentToolbox, max_steps: int = 6):
        self.tpu_loader = tpu_loader
        self.toolbox = toolbox
        self.max_steps = max_steps

    def run_agentic_workflow(self, task_description: str) -> Dict[str, Any]:
        """
        Runs the iterative ReAct loop on TPU.
        """
        print(f"[*] Starting ReAct agent loop on TPU for task: '{task_description}'")
        history = f"{AGENT_SYSTEM_PROMPT}\n\nTask: {task_description}\n"
        
        for step in range(1, self.max_steps + 1):
            print(f"\n--- [AGENT STEP {step}/{self.max_steps}] Prompting DeepSeek-R1-32B ---")
            response = self.tpu_loader.generate(history, max_new_tokens=1024)
            print(f"[Agent Output Preview]:\n{response[:300]}...")
            
            # Check for Action call
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*?)\)", response, re.DOTALL)
            if not action_match:
                # If no action, check if model concluded with JSON
                json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        print("[+] Agent completed investigation and delivered final JSON fleet configuration.")
                        return parsed
                    except Exception:
                        pass
                print("[!] No action found and no final JSON. Terminating loop.")
                break
            
            tool_name = action_match.group(1).strip()
            args_raw = action_match.group(2).strip()
            print(f"[+] Agent decided to execute tool: {tool_name}")
            
            # Execute tool
            obs = self._dispatch_tool(tool_name, args_raw)
            print(f"[Observation Preview]: {str(obs)[:200]}...")
            
            # Append interaction to memory
            history += f"\n{response}\nObservation: {obs}\n"
            
        return {"status": "LOOP_FINISHED", "history_length": len(history)}

    def _dispatch_tool(self, tool_name: str, args_raw: str) -> str:
        """Parses arguments and calls the respective toolbox method."""
        try:
            # Parse simple kwargs
            kwargs = {}
            for match in re.finditer(r'([a-zA-Z0-9_]+)\s*=\s*(".*?"|\'.*?\'|[0-9\.]+)', args_raw):
                k = match.group(1)
                v = match.group(2).strip(' "\'')
                try:
                    if '.' in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    pass
                kwargs[k] = v

            if tool_name == "web_search":
                return self.toolbox.web_search(kwargs.get("query", ""))
            elif tool_name == "fetch_webpage":
                return self.toolbox.fetch_webpage(kwargs.get("url", ""))
            elif tool_name == "query_nim_catalog":
                return self.toolbox.query_nim_catalog()
            elif tool_name == "probe_nim_context_window":
                return self.toolbox.probe_nim_context_window(model_id=kwargs.get("model_id", ""))
            elif tool_name == "test_nim_model":
                return self.toolbox.test_nim_model(
                    model_id=kwargs.get("model_id", ""),
                    temperature=kwargs.get("temperature", 0.6),
                    top_p=kwargs.get("top_p", 0.9),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    system_prompt=kwargs.get("system_prompt", "You are an expert AI."),
                    test_prompt=kwargs.get("test_prompt", "Hello."),
                    reasoning_effort=kwargs.get("reasoning_effort", 100)
                )
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
