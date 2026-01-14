# OpenVista Service Launcher (Windows)
# Quick start script after installation

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  =================================================================" -ForegroundColor Cyan
Write-Host "  |   OpenVista Service Launcher                                 |" -ForegroundColor Cyan
Write-Host "  =================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if services are already running
$backendRunning = $false
$frontendRunning = $false

try {
    $backendResponse = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($backendResponse.StatusCode -eq 200) {
        $backendRunning = $true
        Write-Host "  [OK] Backend is running" -ForegroundColor Green
    }
} catch {
    Write-Host "  [ ] Backend not running" -ForegroundColor Yellow
}

try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($frontendResponse.StatusCode -eq 200) {
        $frontendRunning = $true
        Write-Host "  [OK] Frontend is running" -ForegroundColor Green
    }
} catch {
    Write-Host "  [ ] Frontend not running" -ForegroundColor Yellow
}

Write-Host ""

if ($backendRunning -and $frontendRunning) {
    Write-Host "  All services are running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Opening browser..." -ForegroundColor Cyan
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:3000"
    exit 0
}

# Start backend
if (-not $backendRunning) {
    Write-Host "  Starting backend..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir\backend'; python app.py" -WindowStyle Minimized
    Start-Sleep -Seconds 3
}

# Start frontend
if (-not $frontendRunning) {
    Write-Host "  Starting frontend..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir\frontend'; npm run dev" -WindowStyle Minimized
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "  Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "  [OK] Services started!" -ForegroundColor Green
Write-Host ""
Write-Host "  Opening browser..." -ForegroundColor Cyan
    Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "  Service URLs:" -ForegroundColor Cyan
Write-Host "    Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "    Backend:   http://localhost:5000" -ForegroundColor White
Write-Host "    MaxKB:     http://localhost:8080" -ForegroundColor White
Write-Host ""

