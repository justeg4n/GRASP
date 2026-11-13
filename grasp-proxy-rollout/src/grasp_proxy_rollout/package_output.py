"""Create a compact handoff ZIP from a completed or interrupted SFT run."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_FINAL_FILES = ("adapter_config.json", "adapter_model.safetensors")


def package_output(
    checkpoint_dir: Path,
    output_zip: Path,
    *,
    include_checkpoints: bool,
    allow_partial: bool,
) -> dict[str, object]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_zip = output_zip.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    missing = [name for name in REQUIRED_FINAL_FILES if not (checkpoint_dir / name).is_file()]
    if missing and not allow_partial:
        raise FileNotFoundError(f"Final checkpoint is incomplete; missing: {', '.join(missing)}")
    if output_zip.is_relative_to(checkpoint_dir):
        raise ValueError("Output ZIP must be outside the checkpoint directory")

    validation_path = checkpoint_dir / "checkpoint_validation.json"
    training_manifest_path = checkpoint_dir / "training_manifest.json"
    validation_passed = None
    if validation_path.is_file():
        try:
            validation_passed = (
                json.loads(validation_path.read_text(encoding="utf-8")).get("passed")
                is True
            )
        except (json.JSONDecodeError, OSError):
            validation_passed = False

    files = []
    for path in checkpoint_dir.rglob("*"):
        if not path.is_file() or path.suffix == ".tmp":
            continue
        relative = path.relative_to(checkpoint_dir)
        if not include_checkpoints and any(
            part.startswith("checkpoint-") for part in relative.parts
        ):
            continue
        files.append((path, relative))
    if not files:
        raise ValueError(f"No files found to package in {checkpoint_dir}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    root_name = checkpoint_dir.name
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": str(checkpoint_dir),
        "complete_final_adapter": not missing,
        "training_manifest_present": training_manifest_path.is_file(),
        "checkpoint_validation_present": validation_path.is_file(),
        "checkpoint_validation_passed": validation_passed,
        "handoff_ready": (
            not missing
            and training_manifest_path.is_file()
            and validation_passed is True
        ),
        "missing_required_files": missing,
        "includes_intermediate_checkpoints": include_checkpoints,
        "files": [str(relative).replace("\\", "/") for _, relative in files],
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{root_name}/handoff_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        for source, relative in files:
            archive.write(source, f"{root_name}/{relative.as_posix()}")
    temporary.replace(output_zip)
    return {**manifest, "output_zip": str(output_zip), "bytes": output_zip.stat().st_size}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = package_output(
        args.checkpoint_dir,
        args.output,
        include_checkpoints=args.include_checkpoints,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
