#!/usr/bin/env python3
"""Resilient Colab environment setup for Python 3.13+ without breaking pre-installed PyTorch/CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import urllib.request
import venv
from pathlib import Path

LOCKED_PKGS = ("torch", "torchaudio", "torchvision", "triton")


def create_venv_safely(venv_path: Path) -> None:
    if (venv_path / "bin" / "python").exists():
        print(f"[colab_setup] Reusing existing virtualenv at {venv_path}")
        return
    try:
        subprocess.run(["virtualenv", "--system-site-packages", str(venv_path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[colab_setup] Falling back to EnvBuilder(with_pip=False) + get-pip.py...")
        builder = venv.EnvBuilder(system_site_packages=True, with_pip=False)
        builder.create(str(venv_path))
        get_pip = Path("/tmp/get-pip.py")
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
        subprocess.run([str(venv_path / "bin" / "python"), str(get_pip)], check=True)


def write_constraints(constraints_file: Path) -> None:
    lines = []
    for pkg in LOCKED_PKGS:
        try:
            ver = importlib.metadata.version(pkg)
            lines.append(f"{pkg}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            pass
    constraints_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[colab_setup] Locked runtime constraints:\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=Path("/content/eeg-venv"))
    parser.add_argument("--repo", type=Path, default=Path("/content/code"))
    parser.add_argument("--req", type=str, default="requirements-colab.txt")
    args = parser.parse_args()

    create_venv_safely(args.venv)
    pip_bin = str(args.venv / "bin" / "pip")
    constraints_path = args.venv / "runtime-constraints.txt"
    write_constraints(constraints_path)

    req_file = args.repo / args.req
    if req_file.exists():
        subprocess.run([pip_bin, "install", "-c", str(constraints_path), "-r", str(req_file)], check=True)

    subprocess.run([pip_bin, "install", "-e", str(args.repo), "--no-deps"], check=True)
    print(f"[colab_setup] Ready! Use python: {args.venv / 'bin' / 'python'}")


if __name__ == "__main__":
    main()
