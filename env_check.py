
"""VisionPort Session 4 - Environment Check"""

import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

rows = []


def log(key, value):
    rows.append((key, str(value)))
    print(f"{key:<30}{value}")


def pkg(name):
    try:
        return version(name)
    except Exception:
        return "NOT INSTALLED"


# 1. System and Python
print("\n===== SYSTEM AND PYTHON =====")

log("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))
log("Operating system", platform.platform())
log("Python version", sys.version.split()[0])
log("Python interpreter", sys.executable)


# 2. Package versions
print("\n===== PACKAGE VERSIONS =====")

packages = [
    "torch", "torchvision", "ultralytics", "numpy",
    "pillow", "fastapi", "uvicorn", "streamlit", "pandas"
]

for name in packages:
    log(f"pkg: {name}", pkg(name))


# 3. NVIDIA driver
print("\n===== NVIDIA DRIVER =====")

if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version",
         "--format=csv,noheader"],
        capture_output=True,
        text=True
    )
    log("NVIDIA driver",
        out.stdout.strip().splitlines()[0]
        if out.stdout.strip() else "unreadable")
else:
    log("NVIDIA driver",
        "nvidia-smi not found (normal on this Mac)")


# 4. PyTorch, CUDA, cuDNN and Apple MPS
print("\n===== PYTORCH AND GPU =====")

try:
    import torch

    log("PyTorch version", torch.__version__)
    log("PyTorch CUDA build",
        torch.version.cuda or "None (CPU/MPS build)")
    log("CUDA available", torch.cuda.is_available())

    if torch.backends.cudnn.is_available():
        log("cuDNN version", torch.backends.cudnn.version())
    else:
        log("cuDNN version", "not available")

    # Apple Silicon GPU check
    mps_available = torch.backends.mps.is_available()
    log("Apple MPS available", mps_available)

    if mps_available:
        x = torch.randn(1000, 1000, device="mps")
        y = x @ x
        torch.mps.synchronize()
        log("MPS matmul test", f"PASS (result on {y.device})")
    else:
        log("MPS matmul test", "SKIPPED")

    if torch.cuda.is_available():
        log("GPU count", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log(
                f"GPU {i}",
                f"{p.name}, {p.total_memory / 1e9:.1f} GB"
            )

        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        log("CUDA matmul test", f"PASS (result on {y.device})")
    else:
        log("CUDA matmul test", "SKIPPED (no CUDA GPU)")

except Exception as e:
    log("PyTorch", f"FAILED: {type(e).__name__}: {e}")


# 5. Save environment report
print("\n===== SAVING REPORT =====")

lines = [
    "# VisionPort Environment Report",
    "",
    "| Item | Value |",
    "|---|---|"
]

lines += [f"| {key} | {value} |" for key, value in rows]

Path("env_report.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("\nSaved: env_report.md")
