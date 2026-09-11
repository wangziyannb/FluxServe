import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

OUT = Path(__file__).resolve().parent
MODELS = {
    "fp8": ("thnkinbtfly/llada2.0-flash-fp8", "85bd9f38034aa46f93135676189acec4d7fc40d3"),
    "bf16": ("inclusionAI/LLaDA2.0-flash", "744c3f8c6c8317d2377d6d16d8a3d4be2caef563"),
}

if __name__ == "__main__":
    manifest = {}
    for kind, (repo, revision) in MODELS.items():
        print(f"Downloading {kind}: {repo}@{revision}", flush=True)
        path = snapshot_download(repo, revision=revision, max_workers=8,
                                 allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "*.txt", "*.model"])
        manifest[kind] = {"repo": repo, "revision": revision, "path": path}
        temporary = OUT / "models.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(OUT / "models.json")
        print(f"Ready: {kind} {path}", flush=True)
