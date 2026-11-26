# 启动 U校园 ASR 服务（只监听 127.0.0.1:8000）
#
# 前置（详见 README）：
#   1. 本目录下用 Python 3.12 建好 .venv 并装好 torch/funasr 等
#   2. ffmpeg 在 PATH 中
#   3. SenseVoiceSmall 快照已下载（MODELSCOPE_CACHE 或 UCASR_MODEL_DIR 指向它）
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $root 'src'
$env:PYTHONIOENCODING = 'utf-8'

if (-not $env:MODELSCOPE_CACHE) {
    $env:MODELSCOPE_CACHE = Join-Path $root '.cache\modelscope'
}
if (-not $env:UCASR_MODEL_DIR) {
    $snap = Join-Path $env:MODELSCOPE_CACHE 'models\iic--SenseVoiceSmall\snapshots\master'
    if (Test-Path $snap) { $env:UCASR_MODEL_DIR = $snap }
}

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "找不到 $python —— 请先按 README 用 Python 3.12 创建 .venv 并安装依赖"
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'PATH 里没有 ffmpeg：非 wav 音频会转码失败'
}

& $python -m uvicorn ucasr.main:app --host 127.0.0.1 --port 8000 --log-level warning
