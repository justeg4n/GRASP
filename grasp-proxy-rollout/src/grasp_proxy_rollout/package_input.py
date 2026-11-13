"""Package the exact code and SFT JSONL inputs needed by a Vast.ai run."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT_NAME = "grasp-sft-vast-input"
CODE_FILES = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-gpu.txt",
)
CODE_DIRECTORIES = ("src", "scripts", "docs", "tests")
SFT_FILES = ("train.jsonl", "val.jsonl", "stats.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_generated(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def package_input(repo_dir: Path, sft_dir: Path, output_zip: Path, *, force: bool) -> dict:
    repo_dir = repo_dir.expanduser().resolve()
    sft_dir = sft_dir.expanduser().resolve()
    output_zip = output_zip.expanduser().resolve()
    required = [repo_dir / name for name in CODE_FILES]
    required += [repo_dir / name for name in CODE_DIRECTORIES]
    required += [sft_dir / "sft" / name for name in SFT_FILES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {', '.join(missing)}")
    if output_zip.exists() and not force:
        raise FileExistsError(f"Output exists: {output_zip}; pass --force to replace it")

    entries: list[tuple[Path, str]] = []
    for name in CODE_FILES:
        entries.append((repo_dir / name, f"{ROOT_NAME}/grasp-proxy-rollout/{name}"))
    for name in CODE_DIRECTORIES:
        root = repo_dir / name
        for source in root.rglob("*"):
            if source.is_file() and not _is_generated(source):
                relative = source.relative_to(repo_dir).as_posix()
                entries.append((source, f"{ROOT_NAME}/grasp-proxy-rollout/{relative}"))
    for name in SFT_FILES:
        source = sft_dir / "sft" / name
        entries.append((source, f"{ROOT_NAME}/data/grasp-sft/sft/{name}"))

    data_files = [sft_dir / "sft" / name for name in SFT_FILES]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_source": str(repo_dir),
        "sft_source": str(sft_dir),
        "files": len(entries),
        "sft": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in data_files
        },
    }
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{ROOT_NAME}/input_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        for source, archive_name in entries:
            archive.write(source, archive_name)
    temporary.replace(output_zip)
    return {**manifest, "output_zip": str(output_zip), "zip_bytes": output_zip.stat().st_size}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--sft-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = package_input(args.repo_dir, args.sft_dir, args.output, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
