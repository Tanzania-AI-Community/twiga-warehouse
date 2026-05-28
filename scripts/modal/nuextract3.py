import json
import subprocess
import modal

APP_NAME = "nuextract3-vllm"
MODEL_NAME = "numind/NuExtract3"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=20 * MINUTES,
    scaledown_window=15 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--limit-mm-per-prompt", json.dumps({"image": 99, "video": 0}),
        "--chat-template-content-format", "openai",
        "--generation-config", "vllm",
        "--max-model-len", "131072",
    ]
    print(" ".join(cmd))
    subprocess.Popen(cmd)
