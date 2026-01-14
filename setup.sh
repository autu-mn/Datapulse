#!/bin/bash
# ============================================================
# OpenVista Unified Setup Script (Linux/macOS)
# ============================================================
# One-click installation for OpenVista platform
# - Git LFS large files
# - Docker detection
# - MaxKB deployment
# - GitHub Token configuration
# - Dependencies installation
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Output Functions
# =============================================================================
write_step() {
    echo ""
    echo "[$1] $2"
}

write_success() {
    echo "  [OK] $1"
}

write_fail() {
    echo "  [X] $1"
}

write_info() {
    echo "  -> $1"
}

write_skip() {
    echo "  [SKIP] $1"
}

# =============================================================================
# Welcome Screen
# =============================================================================
show_welcome() {
    clear
    echo ""
    echo "  ================================================================="
    echo "  |                                                               |"
    echo "  |   O P E N V I S T A   Unified Setup Script                   |"
    echo "  |                                                               |"
    echo "  |   GitHub Repository Analytics Platform                       |"
    echo "  |                                                               |"
    echo "  ================================================================="
    echo ""
    
    quotes=(
        "'Code is written for humans to read.' - Harold Abelson"
        "'Make it work, make it right, make it fast.' - Kent Beck"
        "'Simplicity is the ultimate sophistication.' - Leonardo da Vinci"
        "'Talk is cheap. Show me the code.' - Linus Torvalds"
    )
    random_quote=${quotes[$RANDOM % ${#quotes[@]}]}
    echo "  $random_quote"
    echo ""
    echo "  -----------------------------------------------------------------"
    echo ""
    echo "  This script will configure:"
    echo ""
    echo "    [1] Git LFS        (Model weights, training data, database)"
    echo "    [2] Docker          (MaxKB runtime dependency)"
    echo "    [3] MaxKB           (AI Q&A knowledge base)"
    echo "    [4] GitHub Token    (Repository data crawling)"
    echo "    [5] Dependencies    (Python/Node.js packages)"
    echo ""
    echo "  -----------------------------------------------------------------"
    echo ""
    
    read -p "  Press Enter to start, or 'q' to quit: " confirm
    if [ "$confirm" = "q" ] || [ "$confirm" = "Q" ]; then
        echo ""
        echo "  See you next time!"
        echo ""
        exit 0
    fi
}

# =============================================================================
# Step 1: Git LFS
# =============================================================================
install_git_lfs() {
    write_step "1/6" "Git LFS - Large File Storage"
    
    # Check git
    if ! command -v git &> /dev/null; then
        write_fail "Git is not installed"
        write_info "Please install Git first: https://git-scm.com/downloads"
        return 1
    fi
    
    git_version=$(git --version)
    write_info "Git version: $git_version"
    
    # Check git-lfs
    if ! command -v git-lfs &> /dev/null; then
        write_info "Installing Git LFS..."
        
        # Try different package managers
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        elif command -v brew &> /dev/null; then
            brew install git-lfs
        else
            write_fail "Cannot auto-install Git LFS"
            write_info "Please install manually: https://git-lfs.com/"
            return 1
        fi
        
        write_success "Git LFS installed"
    else
        lfs_version=$(git lfs version | head -n 1)
        write_success "Git LFS installed: $lfs_version"
    fi
    
    # Initialize LFS
    write_info "Initializing Git LFS..."
    git lfs install > /dev/null 2>&1
    
    # Check LFS files
    write_info "Checking large files..."
    lfs_files=$(git lfs ls-files 2>&1 || true)
    
    if echo "$lfs_files" | grep -qE "gitpulse_weights.pt|maxkb_full|github_multivar"; then
        write_info "Pulling large files (this may take a few minutes)..."
        echo ""
        echo "    - gitpulse_weights.pt    GitPulse model weights (~100MB)"
        echo "    - github_multivar.json   Training dataset (~500MB)"
        echo "    - maxkb_full.dump        MaxKB knowledge base (~50MB)"
        echo ""
        
        if git lfs pull; then
            write_success "Large files downloaded"
        else
            write_fail "Some files failed. Retry with: git lfs pull"
        fi
    else
        write_success "All large files present"
    fi
    
    # Copy GitPulse config.json to backend/GitPulse/ if needed
    config_source="$SCRIPT_DIR/GitPulse-Training/GitPulse-Model/config.json"
    config_target="$SCRIPT_DIR/backend/GitPulse/config.json"
    
    if [ -f "$config_source" ]; then
        if [ ! -f "$config_target" ]; then
            write_info "Copying GitPulse config.json..."
            mkdir -p "$(dirname "$config_target")"
            cp "$config_source" "$config_target"
            write_success "GitPulse config.json copied"
        else
            write_success "GitPulse config.json already exists"
        fi
    else
        write_info "GitPulse config.json not found at source, skipping copy"
    fi
    
    # Verify GitPulse model files
    weights_file="$SCRIPT_DIR/backend/GitPulse/gitpulse_weights.pt"
    if [ -f "$weights_file" ]; then
        file_size=$(du -m "$weights_file" 2>/dev/null | cut -f1 || echo "0")
        if [ "$file_size" -gt 10 ]; then
            write_success "GitPulse model weights verified ($file_size MB)"
        else
            write_fail "GitPulse model weights file too small ($file_size MB) - may be LFS pointer"
            write_info "Please run 'git lfs pull' to download actual file"
        fi
    else
        write_info "GitPulse model weights not found (will be downloaded when needed)"
    fi
    
    return 0
}

# =============================================================================
# Step 2: Docker
# =============================================================================
install_docker() {
    write_step "2/6" "Docker Environment"
    
    docker_installed=false
    docker_running=false
    
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version)
        docker_installed=true
        write_success "Docker installed: $docker_version"
    else
        write_fail "Docker not installed"
        echo ""
        echo "    Docker is required for MaxKB"
        echo ""
        echo "    Options:"
        echo "    [1] Open Docker installation guide"
        echo "    [2] Skip Docker installation"
        echo ""
        
        read -p "    Enter choice (1/2): " choice
        
        if [ "$choice" = "1" ]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open "https://docs.docker.com/desktop/install/mac-install/"
            else
                xdg-open "https://docs.docker.com/engine/install/" 2>/dev/null || echo "Please visit: https://docs.docker.com/engine/install/"
            fi
            write_info "Please install Docker and rerun this script"
            return 1
        else
            write_skip "Skipping Docker installation"
            return 1
        fi
    fi
    
    # Check if running
    if docker info > /dev/null 2>&1; then
        docker_running=true
        write_success "Docker service is running"
    else
        write_fail "Docker service not running"
        write_info "Please start Docker and rerun this script"
        
        read -p "    Try to start Docker? (y/n): " start_docker
        if [ "$start_docker" = "y" ] || [ "$start_docker" = "Y" ]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open -a Docker
            elif command -v systemctl &> /dev/null; then
                sudo systemctl start docker
            fi
            write_info "Please wait for Docker to start, then rerun this script"
        fi
        return 1
    fi
    
    return 0
}

# =============================================================================
# Step 3: MaxKB
# =============================================================================
install_maxkb() {
    write_step "3/6" "MaxKB Knowledge Base"
    
    BACKUP_FILE="$SCRIPT_DIR/maxkb-export/db/maxkb_full.dump"
    MAXKB_IMAGE="registry.fit2cloud.com/maxkb/maxkb:v2.3.1"
    
    # Check if already installed
    CONTAINER_EXISTS=$(docker ps -a --filter "name=openvista-maxkb" --format "{{.Names}}" 2>/dev/null)
    if [ -n "$CONTAINER_EXISTS" ]; then
        CONTAINER_RUNNING=$(docker ps --filter "name=openvista-maxkb" --format "{{.Names}}" 2>/dev/null)
        if [ -n "$CONTAINER_RUNNING" ]; then
            write_success "MaxKB is installed and running"
            write_info "Access URL: http://localhost:8080"
            
            read -p "    Reinstall? (y/n): " reinstall
            if [ "$reinstall" != "y" ] && [ "$reinstall" != "Y" ]; then
                return 0
            fi
        else
            write_info "MaxKB container exists but not running"
            read -p "    Start? (y) or Reinstall? (r): " start_container
            if [ "$start_container" = "y" ] || [ "$start_container" = "Y" ]; then
                docker start openvista-maxkb > /dev/null 2>&1
                write_success "MaxKB started"
                return 0
            fi
        fi
    fi
    
    # Check backup file
    write_info "Checking data files..."
    if [ ! -f "$BACKUP_FILE" ]; then
        write_fail "Database backup not found: maxkb-export/db/maxkb_full.dump"
        write_info "Please run 'git lfs pull' first"
        return 1
    fi
    
    FILE_SIZE=$(du -m "$BACKUP_FILE" | cut -f1)
    if [ "$FILE_SIZE" -lt 1 ]; then
        write_fail "Backup file too small ($FILE_SIZE MB) - may be LFS pointer"
        write_info "Please run 'git lfs pull' to download actual file"
        return 1
    fi
    write_success "Data file ready ($FILE_SIZE MB)"
    
    # Cleanup
    write_info "Preparing environment..."
    docker stop openvista-maxkb > /dev/null 2>&1 || true
    docker rm -f openvista-maxkb > /dev/null 2>&1 || true
    docker volume create openvista_maxkb_data > /dev/null 2>&1 || true
    docker volume create openvista_maxkb_postgres > /dev/null 2>&1 || true
    
    # Start MaxKB
    write_info "Pulling MaxKB image (may take a while first time)..."
    docker pull "$MAXKB_IMAGE" > /dev/null 2>&1
    
    write_info "Starting MaxKB container..."
    docker run -d --name openvista-maxkb \
        -p 8080:8080 \
        -v openvista_maxkb_data:/opt/maxkb/model \
        -v openvista_maxkb_postgres:/var/lib/postgresql/data \
        -e DB_HOST=localhost \
        -e DB_PORT=5432 \
        -e DB_USER=root \
        -e DB_PASSWORD=MaxKB@123456 \
        -e DB_NAME=maxkb \
        "$MAXKB_IMAGE" > /dev/null 2>&1
    
    # Wait for service
    echo ""
    echo "    Waiting for MaxKB to initialize..."
    
    MAX_WAIT=90
    for i in $(seq 0 $MAX_WAIT); do
        DOTS=$(printf '.%.0s' $(seq 1 $((i % 3 + 1))))
        printf "\r    Initializing%s (%d/%d sec)   " "$DOTS" "$i" "$MAX_WAIT"
        
        if curl -s http://localhost:8080 > /dev/null 2>&1; then
            printf "\r    [OK] MaxKB service ready            \n"
            break
        fi
        sleep 1
    done
    echo ""
    
    # Restore database
    write_info "Restoring knowledge base data..."
    docker cp "$BACKUP_FILE" openvista-maxkb:/tmp/backup.dump > /dev/null 2>&1
    docker exec openvista-maxkb bash -c "psql -U root -d maxkb -c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public;' 2>/dev/null || true" > /dev/null 2>&1
    docker exec openvista-maxkb bash -c "pg_restore -U root -d maxkb --no-owner --no-acl /tmp/backup.dump 2>&1" > /dev/null 2>&1
    docker exec openvista-maxkb rm -f /tmp/backup.dump > /dev/null 2>&1
    
    # Reset password
    write_info "Configuring admin account..."
    PASSWORD_MD5="0df6c52f03e1c75504c7bb9a09c2a016"  # MaxKB@123456 的 MD5 哈希值
    SQL="UPDATE \"user\" SET password = '$PASSWORD_MD5' WHERE username = 'admin';"
    
    # Execute SQL using bash to ensure proper quoting
    docker exec openvista-maxkb bash -c "echo '$SQL' | psql -U root -d maxkb" > /dev/null 2>&1
    
    # Verify password was set correctly
    UPDATE_RESULT=$(docker exec openvista-maxkb bash -c "psql -U root -d maxkb -t -c \"SELECT COUNT(*) FROM \\\"user\\\" WHERE username = 'admin' AND password = '$PASSWORD_MD5';\"" 2>&1 | tr -d '[:space:]')
    
    if [ "$UPDATE_RESULT" = "1" ]; then
        write_success "Admin password reset successfully"
    else
        write_fail "Password reset verification failed"
        write_info "Attempting alternative password reset method..."
        
        # Alternative: Use a SQL file
        TEMP_SQL=$(mktemp)
        echo "$SQL" > "$TEMP_SQL"
        docker cp "$TEMP_SQL" openvista-maxkb:/tmp/reset_password.sql > /dev/null 2>&1
        docker exec openvista-maxkb bash -c "psql -U root -d maxkb -f /tmp/reset_password.sql" > /dev/null 2>&1
        docker exec openvista-maxkb rm -f /tmp/reset_password.sql > /dev/null 2>&1
        rm -f "$TEMP_SQL"
        
        # Verify again
        UPDATE_RESULT2=$(docker exec openvista-maxkb bash -c "psql -U root -d maxkb -t -c \"SELECT COUNT(*) FROM \\\"user\\\" WHERE username = 'admin' AND password = '$PASSWORD_MD5';\"" 2>&1 | tr -d '[:space:]')
        
        if [ "$UPDATE_RESULT2" = "1" ]; then
            write_success "Admin password reset successfully (alternative method)"
        else
            write_fail "Password reset failed. Please run: ./maxkb-export/reset_password.sh"
            write_info "Or manually execute: echo \"UPDATE \\\"user\\\" SET password = '$PASSWORD_MD5' WHERE username = 'admin';\" | docker exec -i openvista-maxkb psql -U root -d maxkb"
        fi
    fi
    
    # Restart
    write_info "Restarting service..."
    docker restart openvista-maxkb > /dev/null 2>&1
    sleep 15
    
    # Wait for service to be fully ready
    write_info "Waiting for service to be ready..."
    MAX_RETRIES=20
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s http://localhost:8080 > /dev/null 2>&1; then
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            write_fail "Service may not be fully ready, but continuing..."
        else
            sleep 3
        fi
    done
    
    write_success "MaxKB installation complete"
    echo ""
    echo "    +-------------------------------------------+"
    echo "    |  MaxKB Knowledge Base Ready               |"
    echo "    |                                           |"
    echo "    |  URL:      http://localhost:8080         |"
    echo "    |  Username: admin                         |"
    echo "    |  Password: MaxKB@123456                   |"
    echo "    +-------------------------------------------+"
    echo ""
    
    return 0
}

# =============================================================================
# Step 4: API Keys Configuration
# =============================================================================
set_api_keys() {
    write_step "4/6" "API Keys Configuration"
    
    env_file="$SCRIPT_DIR/.env"
    github_token=""
    deepseek_key=""
    
    # Check existing config
    if [ -f "$env_file" ]; then
        if grep -q "^GITHUB_TOKEN=" "$env_file"; then
            existing_token=$(grep "^GITHUB_TOKEN=" "$env_file" | cut -d'=' -f2 | tr -d ' ' | tr -d '"')
            if [ -n "$existing_token" ] && [ ${#existing_token} -gt 10 ]; then
                github_token="$existing_token"
                masked_token="${existing_token:0:4}****${existing_token: -4}"
                write_info "Current GitHub Token: $masked_token"
                read -p "    Reconfigure? (y/n): " reconfigure
                if [ "$reconfigure" = "y" ] || [ "$reconfigure" = "Y" ]; then
                    github_token=""
                fi
            fi
        fi
        
        if grep -q "^DEEPSEEK_API_KEY=" "$env_file"; then
            existing_key=$(grep "^DEEPSEEK_API_KEY=" "$env_file" | cut -d'=' -f2 | tr -d ' ' | tr -d '"')
            if [ -n "$existing_key" ] && [ ${#existing_key} -gt 10 ]; then
                deepseek_key="$existing_key"
                masked_key="${existing_key:0:4}****${existing_key: -4}"
                write_info "Current DeepSeek API Key: $masked_key"
                read -p "    Reconfigure? (y/n): " reconfigure
                if [ "$reconfigure" = "y" ] || [ "$reconfigure" = "Y" ]; then
                    deepseek_key=""
                fi
            fi
        fi
    fi
    
    # GitHub Token
    echo ""
    echo "    [1] GitHub Token"
    echo "    Used for:"
    echo "    - Crawling repository Issues, PRs, contributors"
    echo "    - Fetching repository metadata and README"
    echo "    - Higher API quota (5000 requests/hour)"
    echo ""
    echo "    Get Token: https://github.com/settings/tokens"
    echo "    Required scopes: repo (read), read:user"
    echo ""
    
    if [ -z "$github_token" ]; then
        read -p "    Enter GitHub Token (press Enter to skip): " token
        if [ -n "$token" ]; then
            if [ ${#token} -lt 20 ]; then
                write_fail "Invalid token format"
            else
                write_info "Validating token..."
                if curl -s -H "Authorization: token $token" -H "User-Agent: OpenVista-Setup" "https://api.github.com/user" > /dev/null; then
                    user_login=$(curl -s -H "Authorization: token $token" -H "User-Agent: OpenVista-Setup" "https://api.github.com/user" | grep -o '"login":"[^"]*' | cut -d'"' -f4)
                    write_success "Token valid. Logged in as: $user_login"
                    github_token="$token"
                else
                    write_fail "Token validation failed"
                    read -p "    Save this token anyway? (y/n): " continue
                    if [ "$continue" = "y" ] || [ "$continue" = "Y" ]; then
                        github_token="$token"
                    fi
                fi
            fi
        else
            write_skip "Skipping GitHub Token"
            write_info "Note: Without token, crawling is limited (60 requests/hour)"
        fi
    fi
    
    # DeepSeek API Key
    echo ""
    echo "    [2] DeepSeek API Key"
    echo "    Used for:"
    echo "    - AI-powered project summaries"
    echo "    - Intelligent Issue analysis and classification"
    echo "    - Prediction explanations"
    echo ""
    echo "    Get API Key: https://platform.deepseek.com/"
    echo ""
    
    if [ -z "$deepseek_key" ]; then
        read -p "    Enter DeepSeek API Key (press Enter to skip): " key
        if [ -n "$key" ]; then
            if [ ${#key} -lt 20 ]; then
                write_fail "Invalid key format"
            else
                deepseek_key="$key"
                write_success "DeepSeek API Key saved"
            fi
        else
            write_skip "Skipping DeepSeek API Key"
            write_info "Note: AI features will be limited without DeepSeek API Key"
        fi
    fi
    
    # Save to .env
    {
        echo "# OpenVista Environment Configuration"
        echo "# Generated by setup.sh"
        echo ""
        if [ -n "$github_token" ]; then
            echo "# GitHub API Token"
            echo "# Used for repository data crawling"
            echo "GITHUB_TOKEN=$github_token"
            echo ""
        else
            echo "# GitHub API Token"
            echo "# GITHUB_TOKEN=your_token_here"
            echo ""
        fi
        
        if [ -n "$deepseek_key" ]; then
            echo "# DeepSeek API Key"
            echo "# Used for AI summary, Issue analysis, etc."
            echo "DEEPSEEK_API_KEY=$deepseek_key"
        else
            echo "# DeepSeek API Key"
            echo "# DEEPSEEK_API_KEY=your_key_here"
        fi
    } > "$env_file"
    
    write_success "Configuration saved to .env (project root)"
    return 0
}

# =============================================================================
# Step 5: Dependencies
# =============================================================================
install_dependencies() {
    write_step "5/6" "Project Dependencies"
    
    # Python
    write_info "Checking Python environment..."
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        write_success "Python: $python_version"
        
        read -p "    Install backend Python dependencies? (y/n): " install_python
        if [ "$install_python" = "y" ] || [ "$install_python" = "Y" ]; then
            write_info "Installing Python dependencies..."
            cd "$SCRIPT_DIR/backend"
            if pip3 install -r requirements.txt > /dev/null 2>&1; then
                write_success "Python dependencies installed"
            else
                write_fail "Python dependencies installation failed"
            fi
            cd "$SCRIPT_DIR"
        fi
    elif command -v python &> /dev/null; then
        python_version=$(python --version)
        write_success "Python: $python_version"
        
        read -p "    Install backend Python dependencies? (y/n): " install_python
        if [ "$install_python" = "y" ] || [ "$install_python" = "Y" ]; then
            write_info "Installing Python dependencies..."
            cd "$SCRIPT_DIR/backend"
            if pip install -r requirements.txt > /dev/null 2>&1; then
                write_success "Python dependencies installed"
            else
                write_fail "Python dependencies installation failed"
            fi
            cd "$SCRIPT_DIR"
        fi
    else
        write_skip "Python not installed, skipping backend dependencies"
    fi
    
    # Node.js
    write_info "Checking Node.js environment..."
    if command -v node &> /dev/null; then
        node_version=$(node --version)
        write_success "Node.js: $node_version"
        
        read -p "    Install frontend Node.js dependencies? (y/n): " install_node
        if [ "$install_node" = "y" ] || [ "$install_node" = "Y" ]; then
            write_info "Installing Node.js dependencies..."
            cd "$SCRIPT_DIR/frontend"
            if npm install > /dev/null 2>&1; then
                write_success "Node.js dependencies installed"
            else
                write_fail "Node.js dependencies installation failed"
            fi
            cd "$SCRIPT_DIR"
        fi
    else
        write_skip "Node.js not installed, skipping frontend dependencies"
    fi
    
    return 0
}

# =============================================================================
# Step 6: Launch Services
# =============================================================================
start_services() {
    write_step "6/6" "Launch Services"
    
    echo ""
    echo "    Would you like to start all services now?"
    echo ""
    echo "    [1] Start all services (backend, frontend, open browser)"
    echo "    [2] Skip (start manually later)"
    echo ""
    
    read -p "    Enter choice (1/2): " choice
    
    if [ "$choice" != "1" ]; then
        write_skip "Skipping service launch"
        return 0
    fi
    
    # Check if services are already running
    backend_running=false
    frontend_running=false
    
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        backend_running=true
    fi
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        frontend_running=true
    fi
    
    if [ "$backend_running" = true ] && [ "$frontend_running" = true ]; then
        write_success "Services are already running"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "http://localhost:3000"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open "http://localhost:3000" 2>/dev/null || true
        fi
        return 0
    fi
    
    # Check Python
    python_cmd=""
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    elif command -v python &> /dev/null; then
        python_cmd="python"
    fi
    
    # Check Node
    node_available=false
    if command -v node &> /dev/null; then
        node_available=true
    fi
    
    echo ""
    write_info "Starting services..."
    
    # Start backend
    if [ -n "$python_cmd" ] && [ "$backend_running" = false ]; then
        write_info "Starting backend (port 5000)..."
        cd "$SCRIPT_DIR/backend"
        nohup $python_cmd app.py > /dev/null 2>&1 &
        cd "$SCRIPT_DIR"
        sleep 3
    fi
    
    # Start frontend
    if [ "$node_available" = true ] && [ "$frontend_running" = false ]; then
        write_info "Starting frontend (port 3000)..."
        cd "$SCRIPT_DIR/frontend"
        nohup npm run dev > /dev/null 2>&1 &
        cd "$SCRIPT_DIR"
        sleep 5
    fi
    
    # Wait a bit and open browser
    write_info "Waiting for services to start..."
    sleep 8
    
    write_success "Services started"
    write_info "Opening browser..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:3000"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://localhost:3000" 2>/dev/null || true
    fi
    
    return 0
}

# =============================================================================
# Complete
# =============================================================================
show_complete() {
    echo ""
    echo "  ================================================================="
    echo "  |                                                               |"
    echo "  |   Installation Complete!                                    |"
    echo "  |                                                               |"
    echo "  ================================================================="
    echo ""
    echo "  Service URLs:"
    echo ""
    echo "    Frontend:  http://localhost:3000"
    echo "    Backend:   http://localhost:5000"
    echo "    MaxKB:     http://localhost:8080"
    echo ""
    echo "  Quick Start (if services not started):"
    echo ""
    echo "    # Backend"
    echo "    cd backend"
    echo "    python3 app.py"
    echo ""
    echo "    # Frontend (new terminal)"
    echo "    cd frontend"
    echo "    npm run dev"
    echo ""
    echo "  -----------------------------------------------------------------"
    echo ""
    echo "  For more info, see README.md"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
# Temporarily disable exit on error to allow continuing after failures
set +e

show_welcome

install_git_lfs
install_docker
step2=$?

if [ $step2 -eq 0 ]; then
    install_maxkb
else
    write_step "3/6" "MaxKB Knowledge Base"
    write_skip "Docker not available, skipping MaxKB installation"
fi

set_api_keys
install_dependencies
start_services

show_complete

# Re-enable exit on error
set -e

