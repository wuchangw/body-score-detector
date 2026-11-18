$bojiPidPath = Join-Path $PSScriptRoot 'app.pid'
if (-not (Test-Path -LiteralPath $bojiPidPath)) { Write-Output 'No running app recorded.'; exit 0 }
$bojiAppPid = [int](Get-Content -LiteralPath $bojiPidPath -Raw)
$bojiAppPath = Join-Path $PSScriptRoot 'app.py'
$bojiAppProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $bojiAppPid"
if ($bojiAppProcess -and $bojiAppProcess.CommandLine -and $bojiAppProcess.CommandLine.Contains($bojiAppPath)) {
    Stop-Process -Id $bojiAppPid -ErrorAction Stop
    Remove-Item -LiteralPath $bojiPidPath -Force
    Write-Output 'Photo scoring app stopped.'
} else {
    Write-Output 'Recorded process is not this app; no process was stopped.'
}
