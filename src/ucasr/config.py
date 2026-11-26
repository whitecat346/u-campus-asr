"""配置：仅从环境变量读取，默认只监听回环地址（docs/DESIGN.md 决策 7）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def find_model_dir() -> str:
    """按常见位置猜模型快照目录（返回第一个存在的）。"""
    if os.environ.get("UCASR_MODEL_DIR"):
        return os.environ["UCASR_MODEL_DIR"]
    candidates: list[Path] = []
    for cache in (os.environ.get("MODELSCOPE_CACHE"), _REPO_ROOT / ".cache" / "modelscope", _REPO_ROOT.parent / ".cache" / "modelscope"):
        if not cache:
            continue
        base = Path(cache) / "models"
        candidates.append(base / "iic--SenseVoiceSmall" / "snapshots" / "master")
        candidates.append(base / "iic--SenseVoiceSmall" / "snapshots")
    candidates.append(Path.home() / ".cache" / "modelscope" / "models" / "iic--SenseVoiceSmall" / "snapshots")
    for c in candidates:
        if c.is_dir():
            # snapshots/master 或 snapshots/<hash>
            if (c / "model.pt").exists():
                return str(c)
            if c.name == "snapshots":
                sub = sorted(p for p in c.iterdir() if p.is_dir())
                for s in reversed(sub):
                    if (s / "model.pt").exists():
                        return str(s)
    return ""


@dataclass(frozen=True)
class Config:
    host: str = os.environ.get("UCASR_HOST", "127.0.0.1")
    port: int = int(os.environ.get("UCASR_PORT", "8000"))
    model_id: str = os.environ.get("UCASR_MODEL", "iic/SenseVoiceSmall")
    # 模型本地目录（优先于 model_id）；模型缓存目录（避免写 C:\Users\…\.cache 被沙箱拒绝）
    model_dir: str = find_model_dir()
    cache_dir: str = os.environ.get("MODELSCOPE_CACHE", "")
    vad_model: str = os.environ.get("UCASR_VAD", "fsmn-vad")
    device: str = os.environ.get("UCASR_DEVICE", "auto")  # auto | cuda:0 | cpu
    max_upload_mb: int = int(os.environ.get("UCASR_MAX_UPLOAD_MB", "64"))
    max_url_mb: int = int(os.environ.get("UCASR_MAX_URL_MB", "64"))
    download_timeout_s: int = int(os.environ.get("UCASR_DOWNLOAD_TIMEOUT_S", "120"))
    # 启动后在后台把模型加载进显存，首个请求就不用等（设 UCASR_WARM=0 关闭）
    warm: bool = os.environ.get("UCASR_WARM", "1") not in {"0", "false", "False"}
    # 找不到本地快照时是否允许自动下载（900MB 级，默认不允许，避免"跑了很久"）
    allow_download: bool = os.environ.get("UCASR_ALLOW_DOWNLOAD", "0") in {"1", "true", "True"}
    # 识别结果缓存条数（同一段音频重复进入题目时直接命中）
    cache_size: int = int(os.environ.get("UCASR_CACHE_SIZE", "64"))

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch 未装时的兜底
            return "cpu"


CONFIG = Config()
