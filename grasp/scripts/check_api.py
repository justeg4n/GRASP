"""Kiểm tra API teacher trước khi chạy tốn tiền: key, model có tồn tại không, giá hiện hành so với config.

uv run python scripts/check_api.py
uv run python scripts/check_api.py teacher=qwen3_235b_bedrock
uv run python scripts/check_api.py +check_reader=true      # kiểm tra luôn vLLM reader local
"""

import asyncio
import sys

import hydra
from omegaconf import DictConfig

from grasp.llm.budget import CostMeter, Usage
from grasp.llm.client import LLMClient
from grasp.llm.errors import APIFailure
from grasp.utils import script_init


async def run(cfg: DictConfig) -> int:
    import httpx

    tc = cfg.teacher.client
    if not tc.api_key:
        print("[x] chưa đặt API key cho teacher (xem .env.example)")
        return 2
    headers = {"Authorization": f"Bearer {tc.api_key}"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as http:
        r = await http.get(f"{tc.base_url.rstrip('/')}/models")
        if r.status_code in (401, 403):
            print(f"[x] key bị từ chối ({r.status_code}): {r.text[:200]}")
            return 2
        r.raise_for_status()
        models = {m["id"]: m for m in r.json()["data"]}
    if tc.model not in models:
        print(f"[x] model '{tc.model}' không có trên {tc.base_url}. Có: {sorted(models)}")
        return 2
    p = models[tc.model].get("pricing")
    if p:  # FPT trả giá trong /models; Bedrock thì không
        live_in, live_out = float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0)) * 1e6
        print(
            f"[ok] {tc.model}: giá hiện hành ${live_in:.3f}/M in, ${live_out:.3f}/M out "
            f"(config: ${tc.price.input_per_m}/M, ${tc.price.output_per_m}/M)"
        )
        if (
            abs(live_in - tc.price.input_per_m) > 1e-6
            or abs(live_out - tc.price.output_per_m) > 1e-6
        ):
            print(
                "[!] giá trong configs/teacher/*.yaml khác giá hiện hành -> sửa lại để budget tính đúng"
            )
    else:
        print(
            f"[ok] {tc.model}: endpoint không trả giá -> dùng giá trong config "
            f"(${tc.price.input_per_m}/M in, ${tc.price.output_per_m}/M out), tự đối chiếu trang giá"
        )
    if cfg.teacher.get("postpaid"):
        print(
            f"[!] nhà cung cấp trả sau (hết credit sẽ charge thẻ, API KHÔNG báo lỗi): chỉ có "
            f"collect.budget_usd=${cfg.collect.budget_usd}/dataset chặn chi tiêu. Đặt thêm AWS Budgets alert."
        )

    client = LLMClient.from_cfg(tc, meter=CostMeter(), source="teacher")
    usage = Usage()
    try:
        gen = await client.chat(
            "Reply with the single word: pong", max_tokens=256, temperature=0.0, usage=usage
        )
        print(
            f"[ok] teacher trả lời: {gen[0].text.strip()[:80]!r} (finish={gen[0].finish_reason}, "
            f"token in/out={usage.tokens_in}/{usage.tokens_out})"
        )
        if usage.tokens_out > 20:
            print(
                "[!] output token cao bất thường cho 1 từ -> model có reasoning token, cân nhắc tăng max_tokens"
            )
    except APIFailure as e:
        print(f"[x] gọi teacher lỗi: {e}")
        return 2
    finally:
        await client.aclose()

    if cfg.get("check_reader"):
        rc = LLMClient.from_cfg(cfg.reader.client, source="reader", max_retries=1)
        try:
            served = await rc.health()
            ok = cfg.reader.client.model in served
            print(f"[{'ok' if ok else 'x'}] reader {cfg.reader.client.base_url}: {served}")
            if not ok:
                return 2
        except APIFailure as e:
            print(f"[x] reader không phản hồi: {e} (đã chạy scripts/serve_reader.sh chưa?)")
            return 3
        finally:
            await rc.aclose()
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sys.exit(asyncio.run(run(cfg)))


if __name__ == "__main__":
    script_init()
    main()
