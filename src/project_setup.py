
import json, random, subprocess
from pathlib import Path
import numpy as np
import torch

PROJECT_DIR = Path("/content/drive/MyDrive/xray-classifier")
PATHS = {
    "raw": PROJECT_DIR / "data" / "raw",
    "processed": PROJECT_DIR / "data" / "processed",
    "models": PROJECT_DIR / "models",
    "outputs": PROJECT_DIR / "outputs",
    "notebooks": PROJECT_DIR / "notebooks",
    "src": PROJECT_DIR / "src",
    "docs": PROJECT_DIR / "docs",
    "configs": PROJECT_DIR / "configs",
}
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config():
    return json.loads((PATHS["configs"] / "config.json").read_text())

def git_push(message, username, token, repo="xray-classifier"):
    """Commit everything not ignored and push to GitHub."""
    def run(cmd):
        return subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    run(["git", "config", "--global", "--add", "safe.directory", str(PROJECT_DIR)])
    if not (PROJECT_DIR / ".git").exists():
        run(["git", "init", "-b", "main"])
        run(["git", "remote", "add", "origin", f"https://github.com/{username}/{repo}.git"])
    run(["git", "config", "user.name", username])
    run(["git", "config", "user.email", f"{username}@users.noreply.github.com"])
    run(["git", "add", "-A"])
    print(run(["git", "commit", "-m", message]).stdout)
    url = f"https://{username}:{token}@github.com/{username}/{repo}.git"
    out = run(["git", "push", url, "HEAD:main"])
    print((out.stdout + out.stderr).replace(token, "***"))
