from types import SimpleNamespace

import grasp_proxy_rollout.validate_checkpoint as checkpoint_module
from grasp_proxy_rollout.model import GeneratedAction


def _rows(role, target, prompt, count=5):
    return [
        {
            "qid": f"{role}-{index}",
            "role": role,
            "step": index,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
        }
        for index in range(count)
    ]


def test_checkpoint_validation_uses_memory_bounded_batches(monkeypatch):
    samples = {
        "rd": _rows("rd", "Yes (retrieval is required)", "RD prompt"),
        "qf": _rows(
            "qf",
            "Reasoning Chain: None\nKnowledge Gap: missing fact\nQuery: target query",
            "QF prompt",
        ),
        "ki": _rows(
            "ki",
            "<head; relation; tail>",
            "Candidate Triples:\n[1] <head; relation; tail>\n\nQuestion: test?",
        ),
    }
    outputs = {
        16: "Yes (retrieval is required)",
        256: "Reasoning Chain: None\nKnowledge Gap: missing fact\nQuery: target query",
        128: "<head; relation; tail>",
    }

    class FakeProxy:
        checkpoint_kind = "peft_adapter"
        base_model = "Qwen/Qwen2.5-3B-Instruct"

        def __init__(self):
            self.calls = []

        def generate_batch(self, prompts, **kwargs):
            self.calls.append((len(prompts), kwargs))
            text = outputs[kwargs["max_new_tokens"]]
            action = GeneratedAction(text, -0.1, 1, [1], [-0.1])
            return [[action] for _ in prompts]

    proxy = FakeProxy()
    monkeypatch.setattr(checkpoint_module, "_sample_by_role", lambda *args: samples)
    monkeypatch.setattr(
        checkpoint_module.SFTProxy,
        "load",
        lambda checkpoint, load_in_4bit: proxy,
    )
    args = SimpleNamespace(
        samples_per_role=5,
        batch_size=2,
        max_sequence_length=3072,
        prompt_head_tokens=768,
        minimum_format_rate=0.95,
        data_file="val.jsonl",
        seed=42,
        checkpoint="checkpoint",
        load_in_4bit=True,
        output=None,
    )

    report = checkpoint_module.validate_checkpoint(args)

    assert report["passed"] is True
    assert [size for size, _ in proxy.calls] == [2, 2, 1] * 3
    assert {
        call["max_input_tokens"] + call["max_new_tokens"] for _, call in proxy.calls
    } == {3072}
