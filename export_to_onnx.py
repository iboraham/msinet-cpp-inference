# export_to_onnx.py
import os, sys, subprocess, shutil, tempfile
from huggingface_hub import snapshot_download

# 1) Grab the HF model locally
hf_dir = snapshot_download(repo_id="alexanderkroner/MSI-Net")

# 2) Ensure tf2onnx is available
try:
    import tf2onnx  # noqa
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "tf2onnx", "tensorflow==2.*"]
    )

# 3) Convert SavedModel -> ONNX
#   - keep opset moderate (e.g., 13) for broad runtime support
saved_model = hf_dir
onnx_out = "msi_net.onnx"
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        saved_model,
        "--signature_def",
        "serving_default",
        "--output",
        onnx_out,
        "--opset",
        "13",
    ]
)

print("Wrote", onnx_out)
