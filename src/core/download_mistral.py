# src/core/download_mistral_gguf.py

from huggingface_hub import snapshot_download
import os

REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
TARGET_FOLDER = os.path.join("models", "mistral-7b-instruct-v0.2")

print(f"📦 Descargando {REPO_ID} a: {TARGET_FOLDER}")

snapshot_download(
    repo_id=REPO_ID,
    local_dir=TARGET_FOLDER,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("✅ Descarga completada.")
