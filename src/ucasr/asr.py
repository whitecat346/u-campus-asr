"""SenseVoice 推理封装：懒加载单例 + ffmpeg 转码 + 结果归一化。

契约见 docs/DESIGN.md §6。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from .config import CONFIG

_lock = threading.Lock()
_model_singleton: object | None = None


class AsrError(RuntimeError):
    """带错误码的推理异常（FETCH_FAILED / DECODE_FAILED / MODEL_ERROR）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _resolve_model_dir() -> str:
    """把 UCASR_MODEL_DIR / UCASR_MODEL 解析成本地快照目录。

    funasr 1.4.x 的 AutoModel(model=...) 不再接受 "iic/SenseVoiceSmall" 这类
    ModelScope 仓库名（会报 model ... is not registered），必须传本地目录；
    因此这里自己用 modelscope 做一次 snapshot_download。
    """
    if CONFIG.model_dir:
        if not Path(CONFIG.model_dir).exists():
            raise AsrError("MODEL_ERROR", f"UCASR_MODEL_DIR 不存在: {CONFIG.model_dir}")
        return CONFIG.model_dir
    if Path(CONFIG.model_id).exists():
        return CONFIG.model_id
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise AsrError("MODEL_ERROR", f"modelscope 未安装: {exc}") from exc
    cache = CONFIG.cache_dir or None
    try:
        return snapshot_download(CONFIG.model_id, cache_dir=cache)
    except Exception as exc:  # pragma: no cover
        raise AsrError("MODEL_ERROR", f"模型下载失败: {exc}") from exc


def _load_model():
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton
    with _lock:
        if _model_singleton is not None:
            return _model_singleton
        try:
            from funasr import AutoModel
        except Exception as exc:  # pragma: no cover
            raise AsrError("MODEL_ERROR", f"funasr 未安装或导入失败: {exc}") from exc
        model_path = _resolve_model_dir()
        try:
            _model_singleton = AutoModel(
                model=model_path,
                vad_model=CONFIG.vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                device=CONFIG.resolved_device(),
                disable_update=True,
            )
        except Exception as exc:  # pragma: no cover
            raise AsrError("MODEL_ERROR", f"模型加载失败: {exc}") from exc
    return _model_singleton


def is_loaded() -> bool:
    """模型是否已加载（供 /health 使用，不触发加载）。"""
    return _model_singleton is not None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_wav16k(src: Path, dst: Path) -> None:
    """任意容器 → 16k 单声道 wav（SenseVoice 期望的输入）。"""
    if not ffmpeg_available():
        raise AsrError("DECODE_FAILED", "找不到 ffmpeg，无法转码")
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


def recognize_file(path: Path) -> dict:
    """对音频文件做识别，返回契约里的字段。"""
    model = _load_model()
    tmp_dir = Path(tempfile.mkdtemp(prefix="ucasr-"))
    wav = tmp_dir / "in.wav"
    try:
        to_wav16k(path, wav)
        try:
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
        item = _first(rec)
        text = _clean(item.get("text") or "")
        return {
            "text": text,
            "lang": item.get("lang") or "auto",
            "emotion": item.get("emotion"),
            "events": item.get("event") or item.get("events"),
            "duration_ms": int(float(item.get("timestamp", [[0, 0]])[-1][1]) * 1000)
            if item.get("timestamp")
            else None,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_model_json(raw: str) -> dict:
    """SenseVoice 会把富文本塞在 text 里，这里只做 JSON 透传辅助。"""
    try:
        return json.loads(raw)
    except Exception:
        return {}
