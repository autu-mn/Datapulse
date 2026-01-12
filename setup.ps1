<#
.SYNOPSIS
    OpenVista Unified Setup Script
.DESCRIPTION
    One-click installation for OpenVista platform
    - Git LFS large files
    - Docker detection
    - MaxKB deployment
    - GitHub Token configuration
    - Dependencies installation
#>

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# =============================================================================
# Output Functions
# =============================================================================
function Write-Step { 
    param($step, $msg) 
    Write-Host ""
    Write-Host "[$step] " -ForegroundColor Cyan -NoNewline
    Write-Host $msg -ForegroundColor White 
}

function Write-Success { 
    param($msg) 
    Write-Host "  [OK] " -ForegroundColor Green -NoNewline
    Write-Host $msg 
}

function Write-Fail { 
    param($msg) 
    Write-Host "  [X] " -ForegroundColor Red -NoNewline
    Write-Host $msg 
}

function Write-Info { 
    param($msg) 
    Write-Host "  -> " -ForegroundColor Yellow -NoNewline
    Write-Host $msg 
}

function Write-Skip { 
    param($msg) 
    Write-Host "  [SKIP] " -ForegroundColor DarkGray -NoNewline
    Write-Host $msg -ForegroundColor DarkGray 
}

# =============================================================================
# Welcome Screen
# =============================================================================
function Show-Welcome {
    Clear-Host
    Write-Host ""
    Write-Host "  =================================================================" -ForegroundColor Cyan
    Write-Host "  |                                                               |" -ForegroundColor Cyan
    Write-Host "  |   " -ForegroundColor Cyan -NoNewline
    Write-Host "O P E N V I S T A" -ForegroundColor Magenta -NoNewline
    Write-Host "   Unified Setup Script            |" -ForegroundColor Cyan
    Write-Host "  |                                                               |" -ForegroundColor Cyan
    Write-Host "  |   " -ForegroundColor Cyan -NoNewline
    Write-Host "GitHub Repository Analytics Platform" -ForegroundColor White -NoNewline
    Write-Host "                    |" -ForegroundColor Cyan
    Write-Host "  |                                                               |" -ForegroundColor Cyan
    Write-Host "  =================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $quotes = @(
        "'Code is written for humans to read.' - Harold Abelson",
        "'Make it work, make it right, make it fast.' - Kent Beck",
        "'Simplicity is the ultimate sophistication.' - Leonardo da Vinci",
        "'Talk is cheap. Show me the code.' - Linus Torvalds"
    )
    $randomQuote = $quotes | Get-Random
    Write-Host "  $randomQuote" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  -----------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  This script will configure:" -ForegroundColor White
    Write-Host ""
    Write-Host "    [1] Git LFS        " -ForegroundColor Yellow -NoNewline
    Write-Host "(Model weights, training data, database)" -ForegroundColor DarkGray
    Write-Host "    [2] Docker         " -ForegroundColor Yellow -NoNewline
    Write-Host "(MaxKB runtime dependency)" -ForegroundColor DarkGray
    Write-Host "    [3] MaxKB          " -ForegroundColor Yellow -NoNewline
    Write-Host "(AI Q&A knowledge base)" -ForegroundColor DarkGray
    Write-Host "    [4] GitHub Token   " -ForegroundColor Yellow -NoNewline
    Write-Host "(Repository data crawling)" -ForegroundColor DarkGray
    Write-Host "    [5] Dependencies   " -ForegroundColor Yellow -NoNewline
    Write-Host "(Python/Node.js packages)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  -----------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
    
    $confirm = Read-Host "  Press Enter to start, or 'q' to quit"
    if ($confirm -eq "q" -or $confirm -eq "Q") {
        Write-Host "`n  See you next time!`n" -ForegroundColor Cyan
        exit 0
    }
}

# =============================================================================
# Step 1: Git LFS
# =============================================================================
function Install-GitLFS {
    Write-Step "1/6" "Git LFS - Large File Storage"
    
    # Check git
    try {
        $gitVersion = git --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Git not found" }
        Write-Info "Git version: $gitVersion"
    } catch {
        Write-Fail "Git is not installed"
        Write-Info "Please install Git first: https://git-scm.com/downloads"
        return $false
    }
    
    # Check git-lfs
    try {
        $lfsVersion = git lfs version 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Git LFS not found" }
        Write-Success "Git LFS installed: $($lfsVersion.Split(' ')[0..2] -join ' ')"
    } catch {
        Write-Info "Installing Git LFS..."
        
        $installed = $false
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id GitHub.GitLFS -e --silent 2>$null
            $installed = $true
        } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
            choco install git-lfs -y 2>$null
            $installed = $true
        }
        
        if (-not $installed) {
            Write-Fail "Cannot auto-install Git LFS"
            Write-Info "Please install manually: https://git-lfs.com/"
            return $false
        }
        
        Write-Success "Git LFS installed"
    }
    
    # Initialize LFS
    Write-Info "Initializing Git LFS..."
    Push-Location $ScriptDir
    git lfs install 2>&1 | Out-Null
    
    # Check LFS files
    Write-Info "Checking large files..."
    $lfsFiles = git lfs ls-files 2>&1
    
    if ($lfsFiles -match "gitpulse_weights.pt|maxkb_full|github_multivar") {
        Write-Info "Pulling large files (this may take a few minutes)..."
        Write-Host ""
        Write-Host "    - gitpulse_weights.pt    " -ForegroundColor DarkCyan -NoNewline
        Write-Host "GitPulse model weights (~100MB)" -ForegroundColor DarkGray
        Write-Host "    - github_multivar.json   " -ForegroundColor DarkCyan -NoNewline
        Write-Host "Training dataset (~500MB)" -ForegroundColor DarkGray
        Write-Host "    - maxkb_full.sql         " -ForegroundColor DarkCyan -NoNewline
        Write-Host "MaxKB knowledge base (~50MB)" -ForegroundColor DarkGray
        Write-Host ""
        
        git lfs pull 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Large files downloaded"
        } else {
            Write-Fail "Some files failed. Retry with: git lfs pull"
        }
    } else {
        Write-Success "All large files present"
    }
    
    Pop-Location
    return $true
}

# =============================================================================
# Step 2: Docker
# =============================================================================
function Install-Docker {
    Write-Step "2/6" "Docker Environment"
    
    $dockerInstalled = $false
    $dockerRunning = $false
    
    try {
        $dockerVersion = docker --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerInstalled = $true
            Write-Success "Docker installed: $dockerVersion"
        }
    } catch {
        $dockerInstalled = $false
    }
    
    if (-not $dockerInstalled) {
        Write-Fail "Docker not installed"
        Write-Host ""
        Write-Host "    Docker Desktop is required for MaxKB" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    Options:" -ForegroundColor White
        Write-Host "    [1] Open Docker Desktop download page" -ForegroundColor Cyan
        Write-Host "    [2] Skip Docker installation" -ForegroundColor DarkGray
        Write-Host ""
        
        $choice = Read-Host "    Enter choice (1/2)"
        
        if ($choice -eq "1") {
            Start-Process "https://www.docker.com/products/docker-desktop/"
            Write-Info "Please install Docker Desktop and rerun this script"
            Write-Info "Note: Restart required after installation"
            return $false
        } else {
            Write-Skip "Skipping Docker installation"
            return $false
        }
    }
    
    # Check if running
    try {
        docker info 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $dockerRunning = $true
            Write-Success "Docker service is running"
        }
    } catch {
        $dockerRunning = $false
    }
    
    if (-not $dockerRunning) {
        Write-Fail "Docker service not running"
        Write-Info "Please start Docker Desktop and rerun this script"
        
        $startDocker = Read-Host "    Try to start Docker Desktop? (y/n)"
        if ($startDocker -eq "y" -or $startDocker -eq "Y") {
            Write-Info "Starting Docker Desktop..."
            Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -ErrorAction SilentlyContinue
            Write-Info "Please wait for Docker to start, then rerun this script"
        }
        return $false
    }
    
    return $true
}

# =============================================================================
# Step 3: MaxKB
# =============================================================================
function Install-MaxKB {
    Write-Step "3/6" "MaxKB Knowledge Base"
    
    $BackupFile = "$ScriptDir\maxkb-export\db\maxkb_full.dump"
    $MaxKBImage = "registry.fit2cloud.com/maxkb/maxkb"
    
    # Check if already installed
    $containerExists = docker ps -a --filter "name=openvista-maxkb" --format "{{.Names}}" 2>$null
    if ($containerExists) {
        $containerRunning = docker ps --filter "name=openvista-maxkb" --format "{{.Names}}" 2>$null
        if ($containerRunning) {
            Write-Success "MaxKB is installed and running"
            Write-Info "Access URL: http://localhost:8080"
            
            $reinstall = Read-Host "    Reinstall? (y/n)"
            if ($reinstall -ne "y" -and $reinstall -ne "Y") {
                return $true
            }
        } else {
            Write-Info "MaxKB container exists but not running"
            $startContainer = Read-Host "    Start? (y) or Reinstall? (r)"
            if ($startContainer -eq "y" -or $startContainer -eq "Y") {
                docker start openvista-maxkb 2>$null
                Write-Success "MaxKB started"
                return $true
            }
        }
    }
    
    # Check backup file
    Write-Info "Checking data files..."
    if (-not (Test-Path $BackupFile)) {
        Write-Fail "Database backup not found: maxkb-export\db\maxkb_full.dump"
        Write-Info "Please run 'git lfs pull' first"
        return $false
    }
    
    $fileSize = (Get-Item $BackupFile).Length / 1MB
    if ($fileSize -lt 1) {
        Write-Fail "Backup file too small ($([math]::Round($fileSize, 2)) MB) - may be LFS pointer"
        Write-Info "Please run 'git lfs pull' to download actual file"
        return $false
    }
    Write-Success "Data file ready ($([math]::Round($fileSize, 2)) MB)"
    
    # Cleanup
    Write-Info "Preparing environment..."
    docker stop openvista-maxkb 2>$null | Out-Null
    docker rm -f openvista-maxkb 2>$null | Out-Null
    docker volume create openvista_maxkb_data 2>$null | Out-Null
    docker volume create openvista_maxkb_postgres 2>$null | Out-Null
    
    # Start MaxKB
    Write-Info "Pulling MaxKB image (may take a while first time)..."
    docker pull $MaxKBImage 2>&1 | Out-Null
    
    Write-Info "Starting MaxKB container..."
    docker run -d --name openvista-maxkb `
        -p 8080:8080 `
        -v openvista_maxkb_data:/opt/maxkb/model `
        -v openvista_maxkb_postgres:/var/lib/postgresql/data `
        -e DB_HOST=localhost `
        -e DB_PORT=5432 `
        -e DB_USER=root `
        -e DB_PASSWORD=MaxKB@123456 `
        -e DB_NAME=maxkb `
        $MaxKBImage 2>&1 | Out-Null
    
    # Wait for service
    Write-Host ""
    Write-Host "    Waiting for MaxKB to initialize..." -ForegroundColor Yellow
    
    $maxWait = 90
    for ($i = 0; $i -lt $maxWait; $i++) {
        $dots = "." * (($i % 3) + 1)
        Write-Host "`r    Initializing$dots ($i/$maxWait sec)   " -NoNewline -ForegroundColor DarkCyan
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "`r    [OK] MaxKB service ready            " -ForegroundColor Green
                break
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    Write-Host ""
    
    # Restore database
    Write-Info "Restoring knowledge base data..."
    docker cp $BackupFile openvista-maxkb:/tmp/backup.dump 2>&1 | Out-Null
    docker exec openvista-maxkb bash -c "psql -U root -d maxkb -c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public;' 2>/dev/null || true" | Out-Null
    docker exec openvista-maxkb bash -c "pg_restore -U root -d maxkb --no-owner --no-acl /tmp/backup.dump 2>&1" | Out-Null
    docker exec openvista-maxkb rm -f /tmp/backup.dump
    
    # Reset password
    Write-Info "Configuring admin account..."
    $passwordMd5 = "0df6c52f03e1c75504c7bb9a09c2a016"
    $sql = "UPDATE `"user`" SET password = '$passwordMd5' WHERE username = 'admin';"
    echo $sql | docker exec -i openvista-maxkb psql -U root -d maxkb 2>$null | Out-Null
    
    # Restart
    Write-Info "Restarting service..."
    docker restart openvista-maxkb 2>&1 | Out-Null
    Start-Sleep -Seconds 10
    
    Write-Success "MaxKB installation complete"
    Write-Host ""
    Write-Host "    +-------------------------------------------+" -ForegroundColor Green
    Write-Host "    |  MaxKB Knowledge Base Ready               |" -ForegroundColor Green
    Write-Host "    |                                           |" -ForegroundColor Green
    Write-Host "    |  URL:      " -ForegroundColor Green -NoNewline
    Write-Host "http://localhost:8080" -ForegroundColor Cyan -NoNewline
    Write-Host "         |" -ForegroundColor Green
    Write-Host "    |  Username: " -ForegroundColor Green -NoNewline
    Write-Host "admin" -ForegroundColor White -NoNewline
    Write-Host "                         |" -ForegroundColor Green
    Write-Host "    |  Password: " -ForegroundColor Green -NoNewline
    Write-Host "MaxKB@123456" -ForegroundColor White -NoNewline
    Write-Host "                |" -ForegroundColor Green
    Write-Host "    +-------------------------------------------+" -ForegroundColor Green
    Write-Host ""
    
    return $true
}

# =============================================================================
# Step 4: API Keys Configuration
# =============================================================================
function Set-APIKeys {
    Write-Step "4/6" "API Keys Configuration"
    
    $envFile = "$ScriptDir\backend\.env"
    $githubToken = $null
    $deepseekKey = $null
    
    # Check existing config
    if (Test-Path $envFile) {
        $content = Get-Content $envFile -Raw
        if ($content -match "GITHUB_TOKEN=(.+)") {
            $existingToken = $matches[1].Trim()
            if ($existingToken -and $existingToken.Length -gt 10 -and -not $existingToken.StartsWith("#")) {
                $githubToken = $existingToken
            }
        }
        if ($content -match "DEEPSEEK_API_KEY=(.+)") {
            $existingKey = $matches[1].Trim()
            if ($existingKey -and $existingKey.Length -gt 10 -and -not $existingKey.StartsWith("#")) {
                $deepseekKey = $existingKey
            }
        }
    }
    
    # GitHub Token
    Write-Host ""
    Write-Host "    [1] GitHub Token" -ForegroundColor Cyan
    Write-Host "    Used for:" -ForegroundColor White
    Write-Host "    - Crawling repository Issues, PRs, contributors" -ForegroundColor DarkGray
    Write-Host "    - Fetching repository metadata and README" -ForegroundColor DarkGray
    Write-Host "    - Higher API quota (5000 requests/hour)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "    Get Token: " -ForegroundColor Yellow -NoNewline
    Write-Host "https://github.com/settings/tokens" -ForegroundColor Cyan
    Write-Host "    Required scopes: " -ForegroundColor Yellow -NoNewline
    Write-Host "repo (read), read:user" -ForegroundColor DarkGray
    Write-Host ""
    
    if ($githubToken) {
        $maskedToken = $githubToken.Substring(0, 4) + "****" + $githubToken.Substring($githubToken.Length - 4)
        Write-Info "Current: $maskedToken"
        $reconfigure = Read-Host "    Reconfigure? (y/n)"
        if ($reconfigure -eq "y" -or $reconfigure -eq "Y") {
            $githubToken = $null
        }
    }
    
    if (-not $githubToken) {
        $token = Read-Host "    Enter GitHub Token (press Enter to skip)"
        if (-not [string]::IsNullOrWhiteSpace($token)) {
            if ($token.Length -lt 20) {
                Write-Fail "Invalid token format"
            } else {
                Write-Info "Validating token..."
                try {
                    $headers = @{ "Authorization" = "token $token"; "User-Agent" = "OpenVista-Setup" }
                    $response = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers -TimeoutSec 10
                    Write-Success "Token valid. Logged in as: $($response.login)"
                    $githubToken = $token
                } catch {
                    Write-Fail "Token validation failed: $($_.Exception.Message)"
                    $continue = Read-Host "    Save this token anyway? (y/n)"
                    if ($continue -eq "y" -or $continue -eq "Y") {
                        $githubToken = $token
                    }
                }
            }
        } else {
            Write-Skip "Skipping GitHub Token"
            Write-Info "Note: Without token, crawling is limited (60 requests/hour)"
        }
    }
    
    # DeepSeek API Key
    Write-Host ""
    Write-Host "    [2] DeepSeek API Key" -ForegroundColor Cyan
    Write-Host "    Used for:" -ForegroundColor White
    Write-Host "    - AI-powered project summaries" -ForegroundColor DarkGray
    Write-Host "    - Intelligent Issue analysis and classification" -ForegroundColor DarkGray
    Write-Host "    - Prediction explanations" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "    Get API Key: " -ForegroundColor Yellow -NoNewline
    Write-Host "https://platform.deepseek.com/" -ForegroundColor Cyan
    Write-Host ""
    
    if ($deepseekKey) {
        $maskedKey = $deepseekKey.Substring(0, 4) + "****" + $deepseekKey.Substring($deepseekKey.Length - 4)
        Write-Info "Current: $maskedKey"
        $reconfigure = Read-Host "    Reconfigure? (y/n)"
        if ($reconfigure -eq "y" -or $reconfigure -eq "Y") {
            $deepseekKey = $null
        }
    }
    
    if (-not $deepseekKey) {
        $key = Read-Host "    Enter DeepSeek API Key (press Enter to skip)"
        if (-not [string]::IsNullOrWhiteSpace($key)) {
            if ($key.Length -lt 20) {
                Write-Fail "Invalid key format"
            } else {
                $deepseekKey = $key
                Write-Success "DeepSeek API Key saved"
            }
        } else {
            Write-Skip "Skipping DeepSeek API Key"
            Write-Info "Note: AI features will be limited without DeepSeek API Key"
        }
    }
    
    # Save to .env
    $envContent = "# OpenVista Environment Configuration`n"
    $envContent += "# Generated by setup.ps1`n`n"
    
    if ($githubToken) {
        $envContent += "# GitHub API Token`n"
        $envContent += "# Used for repository data crawling`n"
        $envContent += "GITHUB_TOKEN=$githubToken`n`n"
    } else {
        $envContent += "# GitHub API Token`n"
        $envContent += "# GITHUB_TOKEN=your_token_here`n`n"
    }
    
    if ($deepseekKey) {
        $envContent += "# DeepSeek API Key`n"
        $envContent += "# Used for AI summary, Issue analysis, etc.`n"
        $envContent += "DEEPSEEK_API_KEY=$deepseekKey`n"
    } else {
        $envContent += "# DeepSeek API Key`n"
        $envContent += "# DEEPSEEK_API_KEY=your_key_here`n"
    }
    
    # Ensure backend directory exists
    $backendDir = "$ScriptDir\backend"
    if (-not (Test-Path $backendDir)) {
        New-Item -ItemType Directory -Path $backendDir -Force | Out-Null
    }
    
    $envContent | Out-File -FilePath $envFile -Encoding utf8 -NoNewline
    Write-Success "Configuration saved to backend\.env"
    
    return $true
}

# =============================================================================
# Step 5: Dependencies
# =============================================================================
function Install-Dependencies {
    Write-Step "5/6" "Project Dependencies"
    
    # Python
    Write-Info "Checking Python environment..."
    $pythonInstalled = $false
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonInstalled = $true
            Write-Success "Python: $pythonVersion"
        }
    } catch { }
    
    if ($pythonInstalled) {
        $installPython = Read-Host "    Install backend Python dependencies? (y/n)"
        if ($installPython -eq "y" -or $installPython -eq "Y") {
            Write-Info "Installing Python dependencies..."
            Push-Location "$ScriptDir\backend"
            pip install -r requirements.txt 2>&1 | Out-Null
            Pop-Location
            Write-Success "Python dependencies installed"
        }
    } else {
        Write-Skip "Python not installed, skipping backend dependencies"
    }
    
    # Node.js
    Write-Info "Checking Node.js environment..."
    $nodeInstalled = $false
    try {
        $nodeVersion = node --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $nodeInstalled = $true
            Write-Success "Node.js: $nodeVersion"
        }
    } catch { }
    
    if ($nodeInstalled) {
        $installNode = Read-Host "    Install frontend Node.js dependencies? (y/n)"
        if ($installNode -eq "y" -or $installNode -eq "Y") {
            Write-Info "Installing Node.js dependencies..."
            Push-Location "$ScriptDir\frontend"
            npm install 2>&1 | Out-Null
            Pop-Location
            Write-Success "Node.js dependencies installed"
        }
    } else {
        Write-Skip "Node.js not installed, skipping frontend dependencies"
    }
    
    return $true
}

# =============================================================================
# Step 6: Launch Services
# =============================================================================
function Start-Services {
    Write-Step "6/6" "Launch Services"
    
    Write-Host ""
    Write-Host "    Would you like to start all services now?" -ForegroundColor White
    Write-Host ""
    Write-Host "    [1] Start all services (backend, frontend, open browser)" -ForegroundColor Cyan
    Write-Host "    [2] Skip (start manually later)" -ForegroundColor DarkGray
    Write-Host ""
    
    $choice = Read-Host "    Enter choice (1/2)"
    
    if ($choice -ne "1") {
        Write-Skip "Skipping service launch"
        return $false
    }
    
    # Check if services are already running
    $backendRunning = $false
    $frontendRunning = $false
    
    try {
        $backendResponse = Invoke-WebRequest -Uri "http://localhost:5001" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($backendResponse.StatusCode -eq 200) {
            $backendRunning = $true
        }
    } catch { }
    
    try {
        $frontendResponse = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($frontendResponse.StatusCode -eq 200) {
            $frontendRunning = $true
        }
    } catch { }
    
    if ($backendRunning -and $frontendRunning) {
        Write-Success "Services are already running"
        Start-Process "http://localhost:5173"
        return $true
    }
    
    # Check Python
    $pythonAvailable = $false
    try {
        python --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $pythonAvailable = $true
        }
    } catch { }
    
    # Check Node
    $nodeAvailable = $false
    try {
        node --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $nodeAvailable = $true
        }
    } catch { }
    
    Write-Host ""
    Write-Info "Starting services..."
    
    # Start backend
    if ($pythonAvailable -and -not $backendRunning) {
        Write-Info "Starting backend (port 5001)..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir\backend'; python app.py" -WindowStyle Minimized
        Start-Sleep -Seconds 3
    }
    
    # Start frontend
    if ($nodeAvailable -and -not $frontendRunning) {
        Write-Info "Starting frontend (port 5173)..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir\frontend'; npm run dev" -WindowStyle Minimized
        Start-Sleep -Seconds 5
    }
    
    # Wait a bit and open browser
    Write-Info "Waiting for services to start..."
    Start-Sleep -Seconds 8
    
    Write-Success "Services started"
    Write-Info "Opening browser..."
    Start-Process "http://localhost:5173"
    
    return $true
}

# =============================================================================
# Complete
# =============================================================================
function Show-Complete {
    Write-Host ""
    Write-Host "  =================================================================" -ForegroundColor Green
    Write-Host "  |                                                               |" -ForegroundColor Green
    Write-Host "  |   " -ForegroundColor Green -NoNewline
    Write-Host "Installation Complete!" -ForegroundColor White -NoNewline
    Write-Host "                                    |" -ForegroundColor Green
    Write-Host "  |                                                               |" -ForegroundColor Green
    Write-Host "  =================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Service URLs:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    Frontend:  " -ForegroundColor DarkGray -NoNewline
    Write-Host "http://localhost:5173" -ForegroundColor White
    Write-Host "    Backend:   " -ForegroundColor DarkGray -NoNewline
    Write-Host "http://localhost:5001" -ForegroundColor White
    Write-Host "    MaxKB:     " -ForegroundColor DarkGray -NoNewline
    Write-Host "http://localhost:8080" -ForegroundColor White
    Write-Host ""
    Write-Host "  Quick Start (if services not started):" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    # Backend" -ForegroundColor DarkGray
    Write-Host "    cd backend" -ForegroundColor White
    Write-Host "    python app.py" -ForegroundColor White
    Write-Host ""
    Write-Host "    # Frontend (new terminal)" -ForegroundColor DarkGray
    Write-Host "    cd frontend" -ForegroundColor White
    Write-Host "    npm run dev" -ForegroundColor White
    Write-Host ""
    Write-Host "  -----------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  For more info, see README.md" -ForegroundColor DarkGray
    Write-Host ""
}

# =============================================================================
# Main
# =============================================================================
Show-Welcome

$step1 = Install-GitLFS
$step2 = Install-Docker
if ($step2) {
    $step3 = Install-MaxKB
} else {
    Write-Step "3/6" "MaxKB Knowledge Base"
    Write-Skip "Docker not available, skipping MaxKB installation"
    $step3 = $false
}
$step4 = Set-APIKeys
$step5 = Install-Dependencies
$step6 = Start-Services

Show-Complete
