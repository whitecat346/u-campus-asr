"""配置：仅从环境变量读取，默认只监听回环地址（docs/DESIGN.md 决策 7）。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    host: str = os.environ.get("UCASR_HOST", "127.0.0.1")
    port: int = int(os.environ.get("UCASR_PORT", "8000"))
    model_id: str = os.environ.get("UCASR_MODEL", "iic/SenseVoiceSmall")
    vad_model: str = os.environ.get("UCASR_VAD", "fsmn-vad")
    device: str = os.environ.get("UCASR_DEVICE", "auto")  # auto | cuda:0 | cpu
    max_upload_mb: int = int(os.environ.get("UCASR_MAX_UPLOAD_MB", "64"))
    max_url_mb: int = int(os.environ.get("UCASR_MAX_URL_MB", "64"))
    download_timeout_s: int = int(os.environ.get("UCASR_DOWNLOAD_TIMEOUT_S", "120"))

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch 未装时的兜底
            return "cpu"


CONFIG = Config()
