# U校园 ASR 服务端（SenseVoice）

给油猴脚本（[u-campus-assistant](https://github.com/whitecat346/u-campus-assistant)）提供听力题语音识别的本地服务：`FastAPI + FunASR + SenseVoiceSmall`，**只监听 `127.0.0.1:8000`**，浏览器页面（HTTPS）可合法直连回环地址，无混合内容问题。

脚本那边把「ASR 地址」设成 `http://127.0.0.1:8000` 即接上；不部署也不影响判题（听力原文多数随课程内容下发，ASR 只是缺失时的兜底）。

设计契约见 [PAGE-MAP 的服务端章节](https://github.com/whitecat346/u-campus-assistant/blob/main/docs/PAGE-MAP.md)。

获取代码：

```bash
git clone https://github.com/whitecat346/u-campus-asr
cd u-campus-asr
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{status, model, device, ffmpeg, warm, model_dir, loading, cache}` |
| POST | `/asr` | multipart `file`，任意音频（ffmpeg 转 16k 单声道）→ `{text, lang, elapsed_ms, cached}`；同文件二次上传命中 LRU 缓存 |
| POST | `/asr/url` | `{url, headers?}`，服务端下载后识别（拒绝内网地址）；同 URL 二次请求命中缓存 |
| POST | `/cache/clear` | 清空识别结果缓存 |

启动时会按 `UCASR_WARM=1`（默认）在后台线程预热模型，首个请求不必等加载。

## 部署（Windows，本机，**已实测可用**）

前提：本机已有 `uv`、`ffmpeg`（`ffmpeg -version` 能跑即可；本机实测 4.2.2）。

```powershell
cd <本仓库根目录>

# 1) Python 3.12（3.14 装不了 torch，务必钉 3.12）
$env:UV_CACHE_DIR='G:\u_camput_script\.cache\uv'   # 默认 uv 缓存在 C:\Users\… 会被沙箱拒绝：拒绝访问 (os error 5)
uv venv --python 3.12

# 2) 先装 CUDA 版 torch（4060 Ti；cu124 轮子与 torch 2.6 匹配，实测下载约 16 分钟）
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3) 再装其余依赖
uv pip install --python .venv\Scripts\python.exe fastapi uvicorn python-multipart httpx soundfile funasr modelscope
```

```powershell
# 4) 下载模型快照到工作区内（避免写 %USERPROFILE%\.cache 被沙箱拒绝）
$env:MODELSCOPE_CACHE='<仓库根>\.cache\modelscope'
server\.venv\Scripts\python.exe -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('iic/SenseVoiceSmall', cache_dir=r'<仓库根>\.cache\modelscope'))"
# → <仓库根>\.cache\modelscope\models\iic--SenseVoiceSmall\snapshots\master
```

```powershell
# 5) 启动（脚本已封装好 ffmpeg PATH / PYTHONPATH / 模型目录）
pwsh -File G:\u_camput_script\server\run.ps1
```

自测：

```powershell
(Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing).Content
# {"status":"ok","model":"iic/SenseVoiceSmall","device":"cuda:0","ffmpeg":true,"warm":false}
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `UCASR_HOST` | `127.0.0.1` | 不要改成 `0.0.0.0` |
| `UCASR_PORT` | `8000` | |
| `UCASR_MODEL` | `iic/SenseVoiceSmall` | ModelScope 仓库名（服务端会自己 snapshot_download） |
| `UCASR_MODEL_DIR` | 空 | **优先**：指向已下载的本地快照目录 |
| `MODELSCOPE_CACHE` | 空 | 模型缓存目录（建议放工作区内） |
| `UCASR_DEVICE` | `auto` | `cuda:0` / `cpu` |
| `UCASR_MAX_UPLOAD_MB` | `64` | |
| `UCASR_MAX_URL_MB` | `64` | |

## 已知坑（全部已踩过）

- `uv` 默认缓存在 `%USERPROFILE%\scoop\persist\uv\cache`，沙箱下报 `error: Could not acquire lock … 拒绝访问 (os error 5)` → 设 `UV_CACHE_DIR` 到工作区。
- **funasr 1.4.x 的 `AutoModel(model='iic/SenseVoiceSmall')` 不再认这个仓库名**，会报 `model 'iic/SenseVoiceSmall' is not registered`（注册表里只有 `SenseVoiceSmall` 这个类名；传类名又会去下 `modelscope.cn/api/v1/models/SenseVoiceSmall` 而 404）。正确做法：先 `snapshot_download` 拿本地目录，再把目录传给 `AutoModel`——`asr.py` 的 `_resolve_model_dir()` 已经做了这件事。
- ModelScope 默认缓存在 `%USERPROFILE%\.cache\modelscope`，沙箱下 `[WinError 5] 拒绝访问` → 设 `MODELSCOPE_CACHE`。
- `funasr` 1.4.x 的 `generate()` **直接返回 `list[dict]`**（旧版是 `{"result": [...]}`），`_first()` 两种都兼容；返回的 `text` 里混着 `<|en|><|NEUTRAL|><|Speech|><|withitn|>` 富文本标记，`_clean()` 已剥掉。
- 首次推理驻留显存约 2GB，之后常驻（`/health` 的 `warm` 变 `true`）。

## 状态

**已验证可用**（2026-11-26）：`/health` 200、`/asr/url` 对 U校园 CDN 音频（`https://ucontent.cdn.unipus.cn/…_96k.mp3`，免 cookie 可下）返回 1396 字符英文转写；126 秒音频推理约 0.4 秒（rtf 0.003，cuda:0）。


