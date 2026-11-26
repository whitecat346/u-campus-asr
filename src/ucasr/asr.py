"""SenseVoice 推理封装：懒加载单例 + ffmpeg 转码 + 结果归一化 + 结果缓存。

契约见 docs/DESIGN.md §6。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .config import CONFIG, find_model_dir

_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # 同一时刻只跑一次推理，避免显存抖动/并发超时
_model_singleton: object | None = None
_loading = False

_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}
_CACHE_MAX = max(0, CONFIG.cache_size)


class AsrError(RuntimeError):
    """带错误码的推理异常（FETCH_FAILED / DECODE_FAILED / MODEL_ERROR）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def say(msg: str) -> None:
    """统一的进度输出：uvicorn 的 warning 级别会吞掉 info，这里直接写 stderr。"""
    print(f"[ucasr] {msg}", file=sys.stderr, flush=True)


def _resolve_model_dir() -> str:
    """把 UCASR_MODEL_DIR / UCASR_MODEL 解析成本地快照目录。

    funasr 1.4.x 的 AutoModel(model=...) 不再接受 "iic/SenseVoiceSmall" 这类
    ModelScope 仓库名（会报 model ... is not registered），必须传本地目录。
    默认**不**自动下载 900MB 快照（那会让"第一次请求"变成几十分钟），
    只在 UCASR_ALLOW_DOWNLOAD=1 时才下载。
    """
    model_dir = CONFIG.model_dir or find_model_dir()  # 预下载后再探测一次（CONFIG 是 frozen 的）
    if model_dir:
        if not Path(model_dir).exists():
            raise AsrError("MODEL_ERROR", f"UCASR_MODEL_DIR 不存在: {model_dir}")
        return model_dir
    if Path(CONFIG.model_id).exists():
        return CONFIG.model_id
    if not CONFIG.allow_download:
        raise AsrError(
            "MODEL_ERROR",
            "找不到本地模型快照。请先运行 `python -m ucasr.preload` 下载模型"
            "（或在 run.ps1 里加 -Preload），也可以设 UCASR_ALLOW_DOWNLOAD=1 让服务端自动下载。",
        )
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise AsrError("MODEL_ERROR", f"modelscope 未安装: {exc}") from exc
    say(f"正在下载模型快照 {CONFIG.model_id}（约 900MB，可通过 UCASR_MODEL_DIR 复用已有目录）…")
    t0 = time.time()
    try:
        path = snapshot_download(CONFIG.model_id, cache_dir=CONFIG.cache_dir or None)
    except Exception as exc:  # pragma: no cover
        raise AsrError("MODEL_ERROR", f"模型下载失败: {exc}") from exc
    say(f"模型下载完成，用时 {time.time() - t0:.1f}s → {path}")
    return path


def _load_model():
    global _model_singleton, _loading
    if _model_singleton is not None:
        return _model_singleton
    with _load_lock:
        if _model_singleton is not None:
            return _model_singleton
        _loading = True
        try:
            from funasr import AutoModel
        except Exception as exc:  # pragma: no cover
            _loading = False
            raise AsrError("MODEL_ERROR", f"funasr 未安装或导入失败: {exc}") from exc
        model_path = _resolve_model_dir()
        device = CONFIG.resolved_device()
        say(f"加载模型 {model_path}（device={device}）…")
        t0 = time.time()
        try:
            _model_singleton = AutoModel(
                model=model_path,
                vad_model=CONFIG.vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                disable_update=True,
            )
        except Exception as exc:  # pragma: no cover
            raise AsrError("MODEL_ERROR", f"模型加载失败: {exc}") from exc
        finally:
            _loading = False
        say(f"模型就绪，用时 {time.time() - t0:.1f}s")
    return _model_singleton


def preload() -> dict:
    """显式预热（供 `python -m ucasr.preload` 与启动预热使用）。"""
    t0 = time.time()
    model = _load_model()
    load_s = time.time() - t0
    # 用 1 秒静音跑一次，确保 CUDA kernel / VAD 也都初始化（否则首个真实请求仍会慢几秒）
    tmp_dir = Path(tempfile.mkdtemp(prefix="ucasr-warm-"))
    try:
        wav = tmp_dir / "silence.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", str(wav)],
            check=False,
            capture_output=True,
        )
        t1 = time.time()
        with _infer_lock:
            model.generate(input=str(wav), cache={}, language="auto", use_itn=True, batch_size_s=60)
        infer_s = time.time() - t1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    say(f"预热完成：加载 {load_s:.1f}s + 试跑 {infer_s:.2f}s")
    return {"loaded": True, "load_seconds": round(load_s, 2), "probe_seconds": round(infer_s, 2)}


def is_loaded() -> bool:
    """模型是否已加载（供 /health 使用，不触发加载）。"""
    return _model_singleton is not None


def is_loading() -> bool:
    return _loading


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_wav16k(src: Path, dst: Path) -> None:
    """任意容器 → 16k 单声道 wav（SenseVoice 期望的输入）。"""
    if not ffmpeg_available():
        raise AsrError("DECODE_FAILED", "找不到 ffmpeg，无法转码（把 ffmpeg 加进 PATH）")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not dst.exists():
        raise AsrError("DECODE_FAILED", f"ffmpeg 转码失败: {proc.stderr.strip()[:300]}")


_TAG_RE = re.compile(r"<\|[^|]*\|>")


def _clean(text: str) -> str:
    """去掉 SenseVoice 的富文本标记（<|en|> <|NEUTRAL|> <|Speech|> <|withitn|> 等）。"""
    return re.sub(r"\s{2,}", " ", _TAG_RE.sub(" ", text or "")).strip()


def _first(rec):
    """funasr 1.4.x 的 generate() 直接返回 list[dict]；旧版本包一层 {"result": [...]}。"""
    if isinstance(rec, list):
        return rec[0] if rec else {}
    items = rec.get("result") if isinstance(rec, dict) else None
    if isinstance(items, list) and items:
        return items[0]
    return rec if isinstance(rec, dict) else {}


# --------------------------- 结果缓存 ---------------------------


def cache_get(key: str) -> dict | None:
    if not key or _CACHE_MAX == 0:
        return None
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            # 简单 LRU：命中后移到末尾
            _cache.pop(key, None)
            _cache[key] = hit
        return hit


def cache_put(key: str, value: dict) -> None:
    if not key or _CACHE_MAX == 0:
        return
    with _cache_lock:
        _cache.pop(key, None)
        _cache[key] = value
        while len(_cache) > _CACHE_MAX:
            _cache.pop(next(iter(_cache)))


def cache_clear() -> int:
    with _cache_lock:
        n = len(_cache)
        _cache.clear()
        return n


def cache_size() -> int:
    with _cache_lock:
        return len(_cache)


# --------------------------- 推理 ---------------------------


def recognize_file(path: Path) -> dict:
    """对音频文件做识别，返回契约里的字段。"""
    model = _load_model()
    tmp_dir = Path(tempfile.mkdtemp(prefix="ucasr-"))
    wav = tmp_dir / "in.wav"
    try:
        to_wav16k(path, wav)
        t0 = time.time()
        try:
            with _infer_lock:
                rec = model.generate(
                    input=str(wav),
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
        except Exception as exc:  # pragma: no cover
            raise AsrError("MODEL_ERROR", f"推理失败: {exc}") from exc
        elapsed = time.time() - t0
        item = _first(rec)
        text = _clean(item.get("text") or "")
        say(f"识别完成：{len(text)} 字符，耗时 {elapsed:.2f}s")
        return {
            "text": text,
            "lang": item.get("lang") or "auto",
            "emotion": item.get("emotion"),
            "events": item.get("event") or item.get("events"),
            "duration_ms": int(float(item.get("timestamp", [[0, 0]])[-1][1]) * 1000)
            if item.get("timestamp")
            else None,
            "elapsed_ms": int(elapsed * 1000),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_model_json(raw: str) -> dict:
    """SenseVoice 会把富文本塞在 text 里，这里只做 JSON 透传辅助。"""
    try:
        return json.loads(raw)
    except Exception:
        return {}
