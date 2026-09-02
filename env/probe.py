import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import vllm
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def command(args):
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=120)
        return {"returncode": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def disk(path):
    usage = shutil.disk_usage(path)
    return {"path": path, "total_gb": round(usage.total / 2**30, 2), "used_gb": round(usage.used / 2**30, 2), "free_gb": round(usage.free / 2**30, 2)}


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"D0-B requires exactly one visible GPU, found {torch.cuda.device_count()}")

    properties = torch.cuda.get_device_properties(0)
    gpu = {
        "index": 0,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_gb": round(properties.total_memory / 2**30, 2),
    }
    vllm_root = Path(vllm.__file__).resolve().parent
    kernels = {}
    for pattern in ("_C*.so", "_moe_C*.so", "_qutlass_C*.so", "_flashmla_C*.so"):
        for library in sorted(vllm_root.rglob(pattern)):
            sass = command(["cuobjdump", "--list-elf", str(library)])
            ptx = command(["cuobjdump", "--list-ptx", str(library)])
            kernels[str(library)] = {
                "sass_architectures": sorted(set(re.findall(r"sm_[0-9]+", sass["stdout"]))),
                "sass_returncode": sass["returncode"],
                "sass_error": sass["stderr"][-500:],
                "ptx_listing": ptx["stdout"][-2000:],
                "ptx_error": ptx["stderr"][-500:],
            }

    help_result = command(["vllm", "serve", "--help=all"])
    help_text = help_result["stdout"] + "\n" + help_result["stderr"]
    timestamp = datetime.now(timezone.utc)
    config = {"visible_devices": "0", "required_gpu_count": 1, "checks": [
        "versions", "gpu", "disk", "quantization_registry", "attention_registry", "attention_cli", "vllm_kernel_sass"
    ]}
    report = {
        "git_commit": command(["git", "rev-parse", "HEAD"])["stdout"],
        "vllm_version": metadata.version("vllm"),
        "gpu": gpu["name"],
        "timestamp": timestamp.isoformat(),
        "config": config,
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [gpu],
        "prior_two_gpu_measurement": {"p2p": False, "d2d_bandwidth_gbps": 43.6, "source": "AGENTS.md section 2"},
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
        "disks": [disk("/workspace"), disk("/dev/shm")],
        "quantization_methods": sorted(QUANTIZATION_METHODS),
        "attention_backends": sorted(AttentionBackendEnum.__members__),
        "attention_cli": {
            "help_returncode": help_result["returncode"],
            "help_command": "vllm serve --help=all",
            "attention_backend_flag_present": "--attention-backend" in help_text,
            "matching_lines": [line.strip() for line in help_text.splitlines() if "attention" in line.lower()],
        },
        "kernel_architectures": kernels,
    }
    if not report["attention_cli"]["attention_backend_flag_present"]:
        raise SystemExit("vLLM 0.25.1 help does not expose --attention-backend")

    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    output = Path("results") / f"d0b_env_report_{timestamp.strftime('%Y%m%d-%H%M%S')}_{config_hash}.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("attention backend CLI: --attention-backend")


if __name__ == "__main__":
    main()
