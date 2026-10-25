"""Download pinned Dolos-1.5B and Qwen2.5-1.5B-Instruct files into the local HF cache."""

from huggingface_hub import hf_hub_download, snapshot_download

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

if __name__ == "__main__":
    qwen_dir = snapshot_download(QWEN_REPO, revision=QWEN_REVISION, token=False)
    print("qwen:", qwen_dir, flush=True)
    dolos_file = hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False)
    print("dolos:", dolos_file, flush=True)
