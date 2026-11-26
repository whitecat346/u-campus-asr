"""FastAPI 入口：GET /health、POST /asr、POST /asr/url（契约见 docs/DESIGN.md §6）。"""

from __future__ import annotations

import ipaddress
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .asr import AsrError, ffmpeg_available, is_loaded, recognize_file
from .config import CONFIG

app = FastAPI(title="U校园 ASR (SenseVoice)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 回环地址，油猴页面直连
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlRequest(BaseModel):
    url: str = Field(min_length=8)
    headers: dict[str, str] | None = None


def _err(code: str, message: str, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message}})


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": CONFIG.model_id,
        "device": CONFIG.resolved_device(),
        "ffmpeg": ffmpeg_available(),
        "warm": is_loaded(),
    }


@app.post("/asr")
async def asr_upload(file: UploadFile = File(...)) -> dict:
    limit = CONFIG.max_upload_mb * 1024 * 1024
    tmp_dir = Path(tempfile.mkdtemp(prefix="ucasr-up-"))
    try:
        dst = tmp_dir / (file.filename or "audio.bin")
        size = 0
        with dst.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise _err("TOO_LARGE", f"上传超过 {CONFIG.max_upload_mb}MB", 413)
                fh.write(chunk)
        try:
            return recognize_file(dst)
        except AsrError as exc:
            raise _err(exc.code, exc.message, 500) from exc
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/asr/url")
async def asr_url(req: UrlRequest) -> dict:
    parsed = urlparse(req.url)
    if parsed.scheme not in {"http", "https"}:
        raise _err("BAD_URL", "只支持 http/https")
    if _is_loopback_host(parsed.hostname):
        raise _err("BAD_URL", "拒绝访问内网地址")

    limit = CONFIG.max_url_mb * 1024 * 1024
    tmp_dir = Path(tempfile.mkdtemp(prefix="ucasr-url-"))
    try:
        dst = tmp_dir / "download.bin"
        try:
            async with httpx.AsyncClient(timeout=CONFIG.download_timeout_s, follow_redirects=True) as client:
                async with client.stream("GET", req.url, headers=req.headers or {}) as resp:
                    if resp.status_code >= 400:
                        raise _err("FETCH_FAILED", f"下载失败 HTTP {resp.status_code}", 502)
                    size = 0
                    with dst.open("wb") as fh:
                        async for chunk in resp.aiter_bytes(1024 * 256):
                            size += len(chunk)
                            if size > limit:
                                raise _err("TOO_LARGE", f"音频超过 {CONFIG.max_url_mb}MB", 413)
                            fh.write(chunk)
        except (AsrError, HTTPException):
            raise
        except httpx.HTTPError as exc:
            raise _err("FETCH_FAILED", f"下载异常: {exc}", 502) from exc

        try:
            return recognize_file(dst)
        except AsrError as exc:
            raise _err(exc.code, exc.message, 500) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("ucasr.main:app", host=CONFIG.host, port=CONFIG.port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
