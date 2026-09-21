"""Milestone 3 validation runner."""
from __future__ import annotations
import os, subprocess, sys

def main():
    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result=subprocess.run([sys.executable,"-m","pytest","tests","-q"],cwd=root,check=False)
    if result.returncode: return result.returncode
    print("\nMilestone 3 unit validation passed.")
    print("For full inference validation, ensure the trained Milestone 2 emotion model exists.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
