# 启动 SenseVoice ASR 服务（仅监听回环地址 127.0.0.1:8000）
# 用法：pwsh -File server\run.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "找不到 $py，请先按 README 建 venv" }

$env:PATH = "F:\anaconda3\Library\bin;F:\anaconda3\Scripts;" + $env:PATH   # ffmpeg
$env:PYTHONPATH = Join-Path $root 'src'
$env:PYTHONIOENCODING = 'utf-8'
$env:MODELSCOPE_CACHE = 'G:\u_camput_script\.cache\modelscope'
$env:UCASR_MODEL_DIR = 'G:\u_camput_script\.cache\modelscope\models\iic--SenseVoiceSmall\snapshots\master'
$env:UCASR_HOST = '127.0.0.1'
$env:UCASR_PORT = '8000'

& $py -m uvicorn ucasr.main:app --host $env:UCASR_HOST --port $env:UCASR_PORT --log-level warning
