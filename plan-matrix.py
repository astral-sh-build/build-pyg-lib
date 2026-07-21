# /// script
# requires-python = ">=3.13"
# ///
"""Generate the pyg-lib CPU and CUDA build matrix."""

import json
import os


TORCH_BACKENDS = {
    "2.10.0": ["cpu", "cu126", "cu128", "cu130"],
    "2.11.0": ["cpu", "cu126", "cu128", "cu130"],
    "2.12.0": ["cpu", "cu126", "cu130", "cu132"],
    "2.13.0": ["cpu", "cu126", "cu130", "cu132"],
}

TORCH_CUDA_ARCH_LIST = {
    ("2.10", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.10", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.10", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    ("2.11", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.11", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.11", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    ("2.12", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.12", "13.0"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.12", "13.2"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.13", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.13", "13.0"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.13", "13.2"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
}

AUDITWHEEL_EXCLUDES = [
    "libcuda.so",
    "libcuda.so.1",
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_cuda_cpp.so",
    "libtorch_cuda_cu.so",
    "libgomp.so.1",
    "libnvrtc.so",
    "libnvrtc.so.12",
    "libnvrtc.so.13",
    "libcudart.so.12",
    "libcudart.so.13",
]


def main() -> None:
    rows = []
    for torch_version, backends in TORCH_BACKENDS.items():
        for backend in backends:
            torch_minor = ".".join(torch_version.split(".")[:2])
            cuda_version = backend.removeprefix("cu")
            if backend != "cpu":
                cuda_version = f"{cuda_version[:-1]}.{cuda_version[-1]}"

            for target_arch in ("x86_64", "aarch64"):
                rows.append(
                    {
                        "torch-version": torch_version,
                        "torch-minor": torch_minor,
                        "backend": backend,
                        "cuda-version": cuda_version,
                        "target-arch": target_arch,
                        "force-cuda": "0" if backend == "cpu" else "1",
                        "runner": (
                            "depot-ubuntu-24.04-64"
                            if target_arch == "x86_64"
                            else "depot-ubuntu-24.04-arm-64"
                        ),
                        "auditwheel-excludes": " ".join(
                            f"--exclude {library}" for library in AUDITWHEEL_EXCLUDES
                        ),
                        "torch-cuda-arch-list": (
                            ""
                            if backend == "cpu"
                            else TORCH_CUDA_ARCH_LIST[(torch_minor, cuda_version)]
                        ),
                    }
                )

    if os.environ.get("LIMIT_MATRIX") == "1":
        cpu = next(row for row in reversed(rows) if row["backend"] == "cpu")
        cuda = next(row for row in reversed(rows) if row["backend"] != "cpu")
        rows = [cpu, cuda]

    print(json.dumps(rows))


if __name__ == "__main__":
    main()
