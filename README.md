# U校园 ASR 服务端（SenseVoice）

给油猴脚本提供听力题语音识别的本地服务：`FastAPI + FunASR + SenseVoiceSmall`，**只监听 `127.0.0.1:8000`**，浏览器页面（HTTPS）可合法直连回环地址，无混合内容问题。

契约见 [../docs/DESIGN.md](../docs/DESIGN.md) §6。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{status, model, device, ffmpeg, warm}` |
| POST | `/asr` | multipart `file`，任意音频（ffmpeg 转 16k 单声道）→ `{text, lang, emotion, events, duration_ms}` |
| POST | `/asr/url` | `{url, headers?}`，服务端下载后识别（拒绝内网地址） |

## 部署（Windows，本机）

前提：本机已有 `uv`、`ffmpeg`（`G:\...` 环境中 `ffmpeg` 来自 Anaconda，`ffmpeg -version` 可验证）。

```bash
cd server

# 1) Python 3.12（3.14 装不了 torch，务必钉 3.12）
uv python install 3.12
uv venv --python 3.12

# 2) 先装 CUDA 版 torch（4060 Ti；cu124 轮子与 torch>=2.5 匹配）
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3) 再装其余依赖（torch 已满足，不会重复下载）
uv pip install fastapi "uvicorn[standard]" python-multipart httpx funasr modelscope soundfile

# 4) 预热模型（会从 ModelScope 拉 iic/SenseVoiceSmall，约 1GB）
uv run python -c "from funasr import AutoModel; m=AutoModel(model='iic/SenseVoiceSmall', device='cuda:0', disable_update=True); print('model ready')"
```

```bash
# 5) 启动
uv run ucasr                     # 等价于 python -m ucasr
# 或：uv run uvicorn ucasr.main:app --host 127.0.0.1 --port 8000
```

自测：

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@sample.wav" http://127.0.0.1:8000/asr
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `UCASR_HOST` | `127.0.0.1` | 不要改成 `0.0.0.0` |
| `UCASR_PORT` | `8000` | |
| `UCASR_MODEL` | `iic/SenseVoiceSmall` | |
| `UCASR_DEVICE` | `auto` | `cuda:0` / `cpu` |
| `UCASR_MAX_UPLOAD_MB` | `64` | |
| `UCASR_MAX_URL_MB` | `64` | |

## 已知坑

- 本机 shell 的 HTTPS 全崩（`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`）：模型下载若失败，先用 Node 代理拉取或用 `uv` 内置 TLS 重试；ModaScope 下载失败时改用 `modelscope download --model iic/SenseVoiceSmall --local_dir ./models/SenseVoiceSmall`，再把 `UCASR_MODEL` 指向本地目录。
- 首次推理会编译/驻留显存（约 2GB 级别），之后常驻。
- SenseVoice 富文本（情感/事件）以特殊 token 混在 `text` 里；脚本侧只取纯文本。

## 状态

M0：接口与推理封装已落地，**模型尚未装/未验证**（按设计决策：部署排在页面勘察之后）。
