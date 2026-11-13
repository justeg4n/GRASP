import json
from pathlib import Path

from jsonschema import validate

from grasp_proxy_rollout.schema import TRAJECTORY_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OUTPUT = ROOT / "sample_output"


def _read(name):
    return json.loads((SAMPLE_OUTPUT / name).read_text(encoding="utf-8"))


def test_sample_trajectories_match_published_schema():
    schema = _read("trajectory_schema.json")
    assert schema == TRAJECTORY_SCHEMA
    trajectories = _read("trajectories_00000.json")
    assert trajectories
    for trajectory in trajectories:
        validate(trajectory, schema)
        for step in trajectory["steps"]:
            for action in step["proxy_actions"]:
                assert len(action["response_token_ids"]) == len(
                    action["old_token_logprobs"]
                )


def test_sample_group_references_all_trajectories():
    groups = _read("groups.json")
    trajectories = _read("trajectories_00000.json")
    assert groups[0]["trajectory_ids"] == [
        trajectory["trajectory_id"] for trajectory in trajectories
    ]
    assert len({trajectory["group_id"] for trajectory in trajectories}) == 1
