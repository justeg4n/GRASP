"""Reader đóng băng (Qwen2.5-7B-Instruct qua vLLM): sinh từng thought một trong thẻ <think>.

Mỗi bước: prompt = chat_template(user=reader.txt với context HIỆN TẠI) + toàn bộ output trước đó,
dừng ở "</think>". Context thay đổi giữa các bước nhưng các thought trước được giữ nguyên.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from grasp.agents.base import load_template
from grasp.llm.budget import Usage
from grasp.llm.client import LLMClient

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)


def split_thoughts(output: str) -> list[str]:
    return [" ".join(t.split()) for t in _THINK.findall(output or "") if t.strip()]


def extract_answer(output: str) -> str | None:
    m = _ANSWER.search(output or "")
    if not m:
        return None
    ans = " ".join(m.group(1).split()).strip()
    return ans or None


def hf_chat_template(tokenizer_name: str) -> Callable[[str], str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def render(user: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )

    return render


class Reader:
    def __init__(
        self, client: LLMClient, chat_template: Callable[[str], str], max_tokens_answer: int = 1024
    ):
        self.client = client
        self.chat_template = chat_template
        self.max_tokens_answer = max_tokens_answer

    def render(self, question: str, context: str) -> str:
        user = load_template("reader").format(context=context or "None", question=question)
        return self.chat_template(user)

    async def next_thought(
        self, question: str, context: str, output_so_far: str, usage: Usage | None = None
    ) -> str:
        """Trả về đoạn output mới: '<think> ... </think>\\n' hoặc đoạn chứa <answer>."""
        prompt = self.render(question, context) + output_so_far
        gen = await self.client.complete(prompt, stop=["</think>"], usage=usage)
        text = gen.text
        if "<answer>" in text:
            return text if "</answer>" in text else text.rstrip() + " </answer>"
        text = text.rstrip()
        if not text:
            return ""
        if "<think>" not in text:
            text = "<think> " + text.lstrip()
        return text + " </think>\n"

    async def final_answer(self, question: str, context: str, usage: Usage | None = None) -> str:
        """Hết số bước mà chưa có đáp án: sinh lại trọn vẹn với context cuối cùng (như SPARKLE)."""
        gen = await self.client.complete(
            self.render(question, context),
            stop=None,
            max_tokens=self.max_tokens_answer,
            usage=usage,
        )
        return gen.text
