"""Tải file nguồn (HF Hub hoặc URL), có resume khi đứt mạng. File đã đủ thì bỏ qua."""

from __future__ import annotations

import time
from pathlib import Path


def fetch(src: dict, raw_dir: str | Path) -> Path:
    """src: {repo, file, repo_type?} cho HF Hub, hoặc {url, file?} cho URL thường."""
    raw_dir = Path(raw_dir)
    if "repo" in src:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=src["repo"],
                filename=src["file"],
                repo_type=src.get("repo_type", "dataset"),
                local_dir=raw_dir / src["repo"].replace("/", "__"),
            )
        )
    name = src.get("file") or src["url"].rsplit("/", 1)[-1]
    return download_url(src["url"], raw_dir / name)


def download_url(url: str, dest: Path, retries: int = 20) -> Path:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        return dest
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
                if r.status_code == 416:  # đã tải đủ
                    break
                r.raise_for_status()
                total = have + int(r.headers.get("content-length", 0))
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    done = have if mode == "ab" else 0
                    last = time.time()
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            print(f"[download] {dest.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB")
                            last = time.time()
            break
        except (httpx.HTTPError, OSError) as e:
            print(f"[download] lỗi ({e}), thử lại {attempt + 1}/{retries}")
            time.sleep(min(60, 2**attempt))
    else:
        raise RuntimeError(f"không tải được {url}")
    part.replace(dest)
    return dest
