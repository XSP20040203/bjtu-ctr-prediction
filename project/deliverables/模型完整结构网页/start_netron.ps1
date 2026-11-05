$ErrorActionPreference = 'Stop'
$viewerUrl = 'http://127.0.0.1:8765/index.html?view=netron'
$pythonExe = 'C:\Users\30272\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$viewerReady = $false
try {
    $response = Invoke-WebRequest -Uri $viewerUrl -UseBasicParsing -TimeoutSec 3
    $viewerReady = $response.StatusCode -eq 200 -and $response.Content.Contains('Netron')
} catch { }
if (-not $viewerReady) {
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $pythonExe = (Get-Command python -ErrorAction Stop).Source
    }
    Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'http.server', '8765', '--bind', '127.0.0.1', '--directory', ('"' + $PSScriptRoot + '"')) -WindowStyle Hidden
    for ($attempt = 0; $attempt -lt 15; $attempt++) {
        Start-Sleep -Milliseconds 300
        try {
            $response = Invoke-WebRequest -Uri $viewerUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200 -and $response.Content.Contains('Netron')) {
                $viewerReady = $true
                break
            }
        } catch { }
    }
}
if (-not $viewerReady) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show('本地模型查看器未启动。请检查 8765 端口，或在此目录手动启动 Python HTTP 服务。', '模型结构查看器') | Out-Null
    exit 1
}
Start-Process $viewerUrl
