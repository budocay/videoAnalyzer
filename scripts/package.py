#!/usr/bin/env python3
"""Build dist/video-analyzer.zip to hand the project to someone else (standard library only).

Included: code, tests, installers, README/CLAUDE.md, annotations/evaluations/reference images (data/),
and the trained stroke classifier (copied from the local cache to models/, installed by bootstrap.py).
Excluded: .venv, videos and analysis outputs (samples/), annotation pages (regenerable), caches.

Usage: python scripts/package.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "tests", "scripts", "docs", "data/labels", "data/eval", "data/references",
           "install.sh", "install.cmd", "va.sh", "va.cmd", "pyproject.toml", "README.md", "CLAUDE.md", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}
MODEL_CACHE = Path.home() / ".cache" / "video-analyzer" / "models"
EXECUTABLE = {"install.sh", "va.sh", "scripts/bootstrap.py", "scripts/package.py"}


def main() -> None:
    out = ROOT / "dist" / "video-analyzer.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file()) if p.exists() else []
            for f in files:
                rel = f.relative_to(ROOT)
                if SKIP_PARTS & set(rel.parts) or ".egg-info" in str(rel):
                    continue
                info = zipfile.ZipInfo.from_file(f, f"video-analyzer/{rel.as_posix()}")
                if rel.as_posix() in EXECUTABLE:
                    info.external_attr = 0o755 << 16
                z.writestr(info, f.read_bytes(), zipfile.ZIP_DEFLATED)
                n += 1
        for name in ("stroke_clf.pt", "stroke_clf.json"):  # trained on the annotations in data/labels
            if (MODEL_CACHE / name).exists():
                z.write(MODEL_CACHE / name, f"video-analyzer/models/{name}")
                n += 1
    print(f"✔ {out} ({n} fichiers, {out.stat().st_size / 1e6:.1f} Mo)")
    print("À transmettre tel quel : décompresser, puis ./install.sh (macOS/Linux) ou install.cmd (Windows).")


if __name__ == "__main__":
    main()
