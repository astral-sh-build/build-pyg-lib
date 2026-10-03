# build-pyg-lib

Pre-built Linux wheels for [pyg-lib](https://github.com/pyg-team/pyg-lib), the
low-level graph neural network operators used by PyTorch Geometric, across
PyTorch, CUDA, and CPU architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index
for each CUDA version, with CPU-only wheels on the CPU index. Each wheel has a
local version suffix that identifies the accelerator and PyTorch versions it was
built against, such as `pyg-lib==0.9.0+cu.12.8.torch.2.11`, and requires the
matching PyTorch minor release.

Pre-built wheels are available on
[Astral's GPU indexes](https://wheels.astral.sh/index.html). For example, to
install a CUDA 12.8 build:

```console
$ uv add pyg-lib --index astral-cu128=https://wheels.astral.sh/simple/cu128/
```

This configures the index and uses it as the source for `pyg-lib`:

```toml
[tool.uv.sources]
pyg-lib = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astral.sh/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astral.sh/simple/cu128/ pyg-lib
```

For a CPU-only build, use the `https://wheels.astral.sh/simple/cpu/` index
instead.

## GPU tests

The `tests/` directory contains a locked uv project that installs the published
CUDA 12.8 wheel from the Astral index alongside its matching PyTorch build. Run
the tests on a Modal GPU with:

```console
$ modal run tests/modal_app.py
```

Modal installs the locked dependencies in its Linux image and runs the pytest
suite on an NVIDIA A10G. The CUDA wheel is not installed on the local machine.

## Supported versions

Wheels are built from
[pyg-lib 0.9.0](https://github.com/pyg-team/pyg-lib/releases/tag/0.9.0). The
wheels use CPython's stable ABI. PyTorch 2.13 and newer support Python 3.10 through 3.15; older PyTorch builds support Python 3.10 through 3.14.

| PyTorch | Python    | `x86_64` CPU | `aarch64` CPU | `x86_64` CUDA    | `aarch64` CUDA   |
| ------- | --------- | ------------ | ------------- | ---------------- | ---------------- |
| 2.10.0  | 3.10-3.14 | ✓            | ✓             | 12.6, 12.8, 13.0 | 12.6, 12.8, 13.0 |
| 2.11.0  | 3.10-3.14 | ✓            | ✓             | 12.6, 12.8, 13.0 | 12.6, 12.8, 13.0 |
| 2.12.0  | 3.10-3.14 | ✓            | ✓             | 12.6, 13.0, 13.2 | 12.6, 13.0, 13.2 |
| 2.13.0  | 3.10-3.15 | ✓            | ✓             | 12.6, 13.0, 13.2 | 12.6, 13.0, 13.2 |
| 2.14.1  | 3.10-3.15 | ✓            | ✓             | 12.6, 13.0, 13.2 | 12.6, 13.0, 13.2 |

## License

build-pyg-lib is licensed under the [Apache License, Version 2.0](LICENSE).
Built wheels include the upstream and third-party license files alongside Astral
build provenance.

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
