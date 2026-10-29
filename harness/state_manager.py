import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class StateManager:
    """
    Persists and retrieves harness state (active models, verification history, rotation logs).
    Can commit results directly to a GitHub repository or save locally.
    """
    def __init__(self, state_file: str = "active_models.json", history_file: str = "nim_history.log"):
        self.state_file = state_file
        self.history_file = history_file

    def load_local_state(self) -> Dict[str, Any]:
        """Loads state from local JSON file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Warning reading state file: {e}")
        return {"last_updated": None, "active_models": [], "key_status": "UNKNOWN"}

    def save_local_state(self, state_data: Dict[str, Any]) -> None:
        """Saves updated state data locally."""
        state_data["last_updated"] = datetime.now(timezone.utc).isoformat() + "Z"
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        print(f"[+] State successfully saved to {self.state_file}.")

    def append_history_log(self, message: str) -> None:
        """Appends a timestamped line to history log."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def sync_to_github(
        self,
        github_token: str,
        repo_name: str,
        branch: str = "main",
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Pushes updated state to GitHub repository using PyGithub.
        """
        if not github_token or not repo_name:
            print("[!] GitHub token or repo not provided. Skipping remote sync.")
            return False

        try:
            from github import Github
            g = Github(github_token)
            repo = g.get_repo(repo_name)

            commit_msg = commit_message or f"Auto-harness update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
            
            # Read state content
            with open(self.state_file, "r", encoding="utf-8") as f:
                content = f.read()

            try:
                # Update existing file
                contents = repo.get_contents(self.state_file, ref=branch)
                repo.update_file(contents.path, commit_msg, content, contents.sha, branch=branch)
                print(f"[+] Updated {self.state_file} in GitHub repo {repo_name} on branch {branch}.")
            except Exception:
                # Create new file
                repo.create_file(self.state_file, commit_msg, content, branch=branch)
                print(f"[+] Created {self.state_file} in GitHub repo {repo_name} on branch {branch}.")

            return True
        except Exception as e:
            print(f"[-] Failed to sync to GitHub: {e}")
            return False
