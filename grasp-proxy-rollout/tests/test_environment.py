import pytest
from jsonschema import validate

from grasp_proxy_rollout.environment import RolloutEnvironment
from grasp_proxy_rollout.metrics import trajectory_reward
from grasp_proxy_rollout.model import GeneratedAction
from grasp_proxy_rollout.schema import TRAJECTORY_SCHEMA
from grasp_proxy_rollout.trajectory_export import (
    _assign_group_advantages,
    _prepare_output_dir,
)


class FakeProxy:
    def generate_batch(self, prompts, **kwargs):
        prompt = prompts[0]
        if "retrieval decision assistant" in prompt:
            text = "Yes (retrieval is required)"
        elif "query formulation assistant" in prompt:
            text = """Reasoning Chain: None
Knowledge Gap: To continue answering the question, we need to know: population
Query: Lake Grove 2010"""
        else:
            text = "<Lake Grove; population in 2010; 11,163>"
        return [
            [
                GeneratedAction(
                    text=text,
                    mean_logprob=-0.2,
                    token_count=1,
                    token_ids=[42],
                    token_logprobs=[-0.2],
                )
            ]
        ]


class FakeReader:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return (
            "<think>I need census evidence.</think>"
            if self.calls == 1
            else "<answer>11,163</answer>"
        )


class FakeSearcher:
    def search(self, query):
        return [
            {
                "doc_id": "doc-1",
                "title": "Lake Grove",
                "text": "The 2010 population was 11,163.",
                "score": 0.9,
                "triples": [
                    {
                        "head": "Lake Grove",
                        "relation": "population in 2010",
                        "tail": "11,163",
                        "doc_id": "doc-1",
                    }
                ],
            }
        ]


class ImmediateReader:
    def generate(self, prompt):
        return "<answer>11,163</answer>"


class UnusedProxy:
    def generate_batch(self, prompts, **kwargs):
        raise AssertionError("Proxy must not run after the reader answers")


def test_complete_trajectory_contains_rl_handoff_fields():
    environment = RolloutEnvironment(
        FakeProxy(),
        FakeReader(),
        FakeSearcher(),
        max_steps=3,
        proxy_temperature=0.7,
        proxy_top_p=0.95,
        alpha=0.3,
        retrieval_cost=0.0,
        behavior_checkpoint="sft-checkpoint",
    )
    trajectory = environment.run(
        {
            "id": "q1",
            "question": "Population?",
            "gold_answers": ["11,163"],
            "gold_doc_ids": ["doc-1"],
        },
        group_id="q1::group-0",
        trajectory_id="q1::group-0::sample-0",
        sample_index=0,
    )
    assert trajectory["reward"]["total"] == 1.0
    assert trajectory["final_context_doc_ids"] == ["doc-1"]
    assert trajectory["initial_retrieval"]["context_doc_ids"] == ["doc-1"]
    assert trajectory["initial_retrieval"]["counts_toward_retrieval_cost"] is False
    actions = trajectory["steps"][0]["proxy_actions"]
    assert [action["role"] for action in actions] == ["rd", "qf", "ki"]
    assert actions[0]["response_token_ids"] == [42]
    assert actions[0]["old_token_logprobs"] == [-0.2]
    assert actions[2]["selected_candidate_index"] == 0
    trajectory["grpo_advantage"] = 0.0
    validate(trajectory, TRAJECTORY_SCHEMA)


def test_missing_gold_documents_does_not_force_zero_recall():
    reward = trajectory_reward("answer", ["answer"], [], [], 0, 0.3, 0.0)
    assert reward["context_recall"] is None
    assert reward["retrieval_count"] == 0
    assert reward["total"] == 1.0


def test_initial_retrieval_seeds_context_without_counting_as_proxy_retrieval():
    environment = RolloutEnvironment(
        UnusedProxy(),
        ImmediateReader(),
        FakeSearcher(),
        max_steps=3,
        proxy_temperature=0.7,
        proxy_top_p=0.95,
        alpha=0.3,
        retrieval_cost=0.5,
        behavior_checkpoint="sft-checkpoint",
    )
    trajectory = environment.run(
        {
            "id": "q1",
            "question": "Population?",
            "gold_answers": ["11,163"],
            "gold_doc_ids": ["doc-1"],
        },
        group_id="g",
        trajectory_id="t",
        sample_index=0,
    )
    assert trajectory["reward"]["retrieval_count"] == 0
    assert trajectory["reward"]["retrieval_cost"] == 0.0
    assert trajectory["reward"]["total"] == 1.0
    assert trajectory["steps"][0]["context_before_doc_ids"] == ["doc-1"]


def test_group_advantage_is_attached_to_each_trajectory():
    trajectories = [
        {
            "group_id": "g",
            "question_id": "q",
            "trajectory_id": f"t-{index}",
            "reward": {"total": reward},
            "grpo_advantage": None,
        }
        for index, reward in enumerate([0.0, 1.0])
    ]
    group = _assign_group_advantages(trajectories)
    assert trajectories[0]["grpo_advantage"] < 0
    assert trajectories[1]["grpo_advantage"] > 0
    assert group["zero_variance"] is False


def test_rollout_output_directory_requires_explicit_overwrite(tmp_path):
    managed = tmp_path / "manifest.json"
    unmanaged = tmp_path / "notes.txt"
    managed.write_text("old", encoding="utf-8")
    unmanaged.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare_output_dir(tmp_path, overwrite=False)
    _prepare_output_dir(tmp_path, overwrite=True)
    assert not managed.exists()
    assert unmanaged.read_text(encoding="utf-8") == "keep"
