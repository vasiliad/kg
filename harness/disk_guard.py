"""
Proactive Disk Guard & Storage Janitor.
Guards against disk quota exhaustion, log bloat, cache accumulation,
and Git packfile inflation across Kaggle TPU, GitHub Actions CPU, and local environments.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

class DiskGuard:
    def __init__(self, target_dir: str = "."):
        self.target_dir = Path(target_dir).resolve()

    def inspect_disk_usage(self) -> Dict[str, Any]:
        """
        Returns accurate disk metrics for the current execution volume.
        """
        usage = shutil.disk_usage(str(self.target_dir))
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        percent_used = (usage.used / usage.total) * 100.0

        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "percent_used": round(percent_used, 1),
            "is_critical": free_gb < 1.0 or percent_used > 92.0,
            "is_warning": free_gb < 2.5 or percent_used > 80.0
        }

    def prune_temporary_bloat(self) -> Dict[str, Any]:
        """
        Safely prunes garbage files, python caches, and temporary artifacts.
        """
        reclaimed_bytes = 0
        removed_items = []

        # 1. Clean __pycache__ directories
        for root, dirs, files in os.walk(str(self.target_dir)):
            for d in list(dirs):
                if d == "__pycache__" or d == ".ipynb_checkpoints":
                    full_path = os.path.join(root, d)
                    try:
                        shutil.rmtree(full_path, ignore_errors=True)
                        removed_items.append(d)
                    except Exception:
                        pass

        # 2. Clean temporary files
        temp_patterns = [".tmp", ".log.bak", ".log.old", ".pytest_cache"]
        for root, dirs, files in os.walk(str(self.target_dir)):
            for f in files:
                if any(f.endswith(pat) for pat in temp_patterns) or f.startswith("tmp_"):
                    full_path = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(full_path)
                        os.remove(full_path)
                        reclaimed_bytes += sz
                        removed_items.append(f)
                    except Exception:
                        pass

        # 3. Clean Hugging Face temp lock files if present
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if os.path.exists(hf_home):
            for root, dirs, files in os.walk(hf_home):
                for f in files:
                    if f.endswith(".incomplete") or f.endswith(".lock"):
                        full_path = os.path.join(root, f)
                        try:
                            os.remove(full_path)
                            removed_items.append(f)
                        except Exception:
                            pass

        # 4. Optional Git GC (auto-repack loose objects)
        git_dir = self.target_dir / ".git"
        if git_dir.exists():
            try:
                subprocess.run(["git", "gc", "--auto", "--quiet"], cwd=str(self.target_dir), check=False)
            except Exception:
                pass

        return {
            "reclaimed_mb": round(reclaimed_bytes / (1024 ** 2), 2),
            "removed_count": len(removed_items)
        }

    def enforce_quota_guard(self) -> Dict[str, Any]:
        """
        Comprehensive pre-flight check executed at the start of any harness cycle.
        """
        before = self.inspect_disk_usage()
        
        prune_res = {}
        if before["is_warning"] or before["is_critical"]:
            print(f"[DiskGuard] Warning: Storage threshold reached ({before['percent_used']}% used, {before['free_gb']} GB free). Running janitor prune...")
            prune_res = self.prune_temporary_bloat()
            after = self.inspect_disk_usage()
            print(f"[DiskGuard] Prune completed: Reclaimed {prune_res.get('reclaimed_mb')} MB. New free space: {after['free_gb']} GB.")
        else:
            after = before

        return {
            "status": "DANGER_ABORT" if after["is_critical"] else "HEALTHY",
            "metrics": after,
            "prune_summary": prune_res
        }

if __name__ == "__main__":
    dg = DiskGuard()
    report = dg.enforce_quota_guard()
    print("DiskGuard Status Report:", report)
