"""模型预下载 + 预热：`python -m ucasr.preload [--download] [--no-warm]`

用途：把「第一次请求要等十分钟」提前到部署阶段完成。
- 默认只做本地快照校验与加载（不联网）；
- `--download`：本地没有快照时用 modelscope 下载到 MODELSCOPE_CACHE；
- `--no-warm`：只下载/校验，不加载模型到显存。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .config import CONFIG


def _download_if_needed() -> str:
    if CONFIG.model_dir and Path(CONFIG.model_dir).exists():
        return CONFIG.model_dir
    from modelscope.hub.snapshot_download import snapshot_download

    cache = CONFIG.cache_dir or str(Path.cwd() / ".cache" / "modelscope")
    os.environ.setdefault("MODELSCOPE_CACHE", cache)
    print(f"[ucasr] 正在下载 {CONFIG.model_id} 到 {cache}（约 900MB，请耐心等）…", flush=True)
    t0 = time.time()
    path = snapshot_download(CONFIG.model_id, cache_dir=cache)
    print(f"[ucasr] 下载完成，用时 {time.time() - t0:.1f}s → {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="预下载 / 预热 U校园 ASR 模型")
    ap.add_argument("--download", action="store_true", help="本地没有快照时自动下载")
    ap.add_argument("--no-warm", action="store_true", help="只校验/下载，不加载模型")
    args = ap.parse_args()

    import shutil

    print(f"[ucasr] model_id={CONFIG.model_id}", flush=True)
    print(f"[ucasr] model_dir={CONFIG.model_dir or '(未找到)'}", flush=True)
    print(
        f"[ucasr] device={CONFIG.resolved_device()}  ffmpeg={shutil.which('ffmpeg') or '(未找到，非 wav 音频会失败)'}",
        flush=True,
    )

    if not CONFIG.model_dir and not args.download:
        print(
            "[ucasr] 没找到本地快照。加 --download 下载，或先设置 UCASR_MODEL_DIR 指向已有目录。",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if args.download:
        _download_if_needed()

    if args.no_warm:
        print("[ucasr] 跳过加载（--no-warm）", flush=True)
        return 0

    from .asr import preload

    preload()
    print("[ucasr] 预热完成，可以直接启动服务了（run.ps1 或 uvicorn）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
