import json
import zipfile

import pytest

from grasp_proxy_rollout.package_input import package_input
from grasp_proxy_rollout.package_output import package_output


def test_package_input_contains_code_data_and_manifest(tmp_path):
    repo = tmp_path / "repo"
    sft = tmp_path / "artifact" / "sft"
    for name in ("src", "scripts", "docs", "tests"):
        (repo / name).mkdir(parents=True)
        (repo / name / "kept.txt").write_text(name, encoding="utf-8")
    for name in (
        ".gitattributes",
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-gpu.txt",
    ):
        (repo / name).write_text(name, encoding="utf-8")
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "ignored.pyc").write_bytes(b"generated")
    sft.mkdir(parents=True)
    for name in ("train.jsonl", "val.jsonl", "stats.json"):
        (sft / name).write_text(name, encoding="utf-8")
    output = tmp_path / "input.zip"

    report = package_input(repo, sft.parent, output, force=False)

    assert report["files"] == 13
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "grasp-sft-vast-input/input_manifest.json" in names
        assert "grasp-sft-vast-input/data/grasp-sft/sft/train.jsonl" in names
        assert "grasp-sft-vast-input/grasp-proxy-rollout/src/kept.txt" in names
        assert not any("__pycache__" in name for name in names)


def test_package_output_contains_final_adapter_logs_and_no_intermediate_checkpoint(tmp_path):
    checkpoint = tmp_path / "proxy-sft"
    (checkpoint / "logs").mkdir(parents=True)
    (checkpoint / "checkpoint-250").mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "checkpoint_validation.json").write_text(
        '{"passed": true}', encoding="utf-8"
    )
    (checkpoint / "training_manifest.json").write_text("{}", encoding="utf-8")
    (checkpoint / "logs" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (checkpoint / "checkpoint-250" / "optimizer.pt").write_bytes(b"optimizer")
    output = tmp_path / "proxy-sft-handoff.zip"

    report = package_output(
        checkpoint,
        output,
        include_checkpoints=False,
        allow_partial=False,
    )

    assert report["complete_final_adapter"] is True
    assert report["training_manifest_present"] is True
    assert report["handoff_ready"] is True
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "proxy-sft/adapter_model.safetensors" in names
        assert "proxy-sft/logs/metrics.jsonl" in names
        assert "proxy-sft/checkpoint-250/optimizer.pt" not in names
        manifest = json.loads(archive.read("proxy-sft/handoff_manifest.json"))
        assert manifest["complete_final_adapter"] is True
        assert manifest["training_manifest_present"] is True
        assert manifest["checkpoint_validation_passed"] is True


def test_package_output_rejects_incomplete_final_adapter(tmp_path):
    checkpoint = tmp_path / "proxy-sft"
    checkpoint.mkdir()
    (checkpoint / "console.log").write_text("failed", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        package_output(
            checkpoint,
            tmp_path / "output.zip",
            include_checkpoints=False,
            allow_partial=False,
        )


def test_package_output_without_validation_is_not_handoff_ready(tmp_path):
    checkpoint = tmp_path / "proxy-sft"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "training_manifest.json").write_text("{}", encoding="utf-8")

    report = package_output(
        checkpoint,
        tmp_path / "handoff.zip",
        include_checkpoints=False,
        allow_partial=False,
    )

    assert report["complete_final_adapter"] is True
    assert report["checkpoint_validation_present"] is False
    assert report["handoff_ready"] is False
