"""Paste into a Kaggle notebook cell (GPU on). Attach datasets, then run.

    !git clone <repo> /kaggle/working/VidCap && cd /kaggle/working/VidCap
    !python -m scripts.kaggle_phase0

Re-running after a 12h session kill resumes: every cached shard is skipped.
Copy /kaggle/working/cache out as a Kaggle Dataset so later phases never re-embed.
"""
import subprocess
import sys

# Kaggle mounts each attached dataset under its own /kaggle/input/<slug>/ — map slug to our name.
ROOTS = {
    "msrvtt": "/kaggle/input/msrvtt",
    "tvsum": "/kaggle/input/tvsum",
    "summe": "/kaggle/input/summe",
    "activitynet": "/kaggle/input/activitynet",
    "holdout": "/kaggle/input/vidcap-holdout",
}

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "scripts.smoke_test"], check=True)
    for name, root in ROOTS.items():
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, "-m", "scripts.build_cache", name, "--root", root])
        if r.returncode:
            print(f"  skipped {name} (not attached or empty)")
