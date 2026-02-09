#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════
# AI Trader - Cloud Deployment Script
# Run this on a fresh Ubuntu 24.04 VPS
#
# Usage:
#   curl -sSL <raw-url>/scripts/deploy.sh | bash
#   OR
#   git clone <repo> && cd Trader && bash scripts/deploy.sh
# ═══════════════════════════════════════════════════════════

REPO_URL="https://github.com/asvweeren/Personal-Trader.git"
INSTALL_DIR="$HOME/trader"

echo "========================================="
echo "  AI Trader - Cloud Deployment"
echo "========================================="

# ── Step 1: Install Docker ─────────────────────────────────
install_docker() {
    if command -v docker &> /dev/null; then
        echo "[OK] Docker already installed"
    else
        echo "[..] Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER"
        echo "[OK] Docker installed. You may need to log out and back in for group changes."
    fi

    if command -v docker compose &> /dev/null; then
        echo "[OK] Docker Compose available"
    else
        echo "[!!] Docker Compose not found. Install it manually."
        exit 1
    fi
}

# ── Step 2: Clone repo ────────────────────────────────────
clone_repo() {
    if [ -d "$INSTALL_DIR" ]; then
        echo "[..] Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull origin master
    else
        echo "[..] Cloning repository..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
}

# ── Step 3: Configure environment ─────────────────────────
configure_env() {
    if [ -f "$INSTALL_DIR/.env" ]; then
        echo "[OK] .env already exists - skipping"
        echo "     Edit it manually if needed: nano $INSTALL_DIR/.env"
    else
        echo "[..] Creating .env from template..."
        cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"

        # Generate random secrets
        SECRET_KEY=$(openssl rand -hex 32)
        PG_PASS=$(openssl rand -hex 16)
        VNC_PASS=$(openssl rand -hex 8)

        sed -i "s/change-me-to-a-random-string/$SECRET_KEY/" "$INSTALL_DIR/.env"
        sed -i "s/CHANGE_ME_strong_password_here/$PG_PASS/" "$INSTALL_DIR/.env"
        sed -i "s/change_me_vnc_password/$VNC_PASS/" "$INSTALL_DIR/.env"

        echo ""
        echo "  ╔══════════════════════════════════════════╗"
        echo "  ║  IMPORTANT: Edit .env before continuing  ║"
        echo "  ╚══════════════════════════════════════════╝"
        echo ""
        echo "  You MUST set these values:"
        echo "    - IBKR_USERNAME     (Interactive Brokers username)"
        echo "    - IBKR_PASSWORD     (Interactive Brokers password)"
        echo "    - ANTHROPIC_API_KEY (Claude API key)"
        echo "    - NEWS_API_KEY      (NewsAPI.org key)"
        echo "    - DOMAIN            (your domain, for SSL)"
        echo ""
        echo "  Optional but recommended:"
        echo "    - SMTP_* settings   (email alerts)"
        echo "    - SIGNAL_*          (Signal messenger alerts)"
        echo ""
        echo "  Edit now:  nano $INSTALL_DIR/.env"
        echo ""
        read -p "  Press Enter when done editing .env..."
    fi
}

# ── Step 4: Setup SSL ─────────────────────────────────────
setup_ssl() {
    source "$INSTALL_DIR/.env"

    if [ "$DOMAIN" = "trader.yourdomain.com" ] || [ -z "$DOMAIN" ]; then
        echo "[!!] No domain configured - skipping SSL setup"
        echo "     The app will be available on HTTP port 80"
        echo "     Set DOMAIN in .env and re-run to enable SSL"
        return
    fi

    echo "[..] Setting up SSL for $DOMAIN..."

    # Start nginx temporarily for the ACME challenge
    docker compose -f docker-compose.prod.yml up -d nginx 2>/dev/null || true

    # Request certificate
    docker compose -f docker-compose.prod.yml run --rm certbot \
        certbot certonly --webroot \
        --webroot-path=/var/www/certbot \
        --email "$SMTP_USER" \
        --agree-tos \
        --no-eff-email \
        -d "$DOMAIN" \
        --cert-name trader

    if [ $? -eq 0 ]; then
        echo "[OK] SSL certificate obtained"
        echo "[..] Switching to SSL nginx config..."
        # Replace nginx config with SSL version
        cp "$INSTALL_DIR/nginx/nginx-ssl.conf" "$INSTALL_DIR/nginx/nginx.conf"
        docker compose -f docker-compose.prod.yml restart nginx
        echo "[OK] SSL enabled"
    else
        echo "[!!] SSL setup failed - continuing with HTTP"
    fi
}

# ── Step 5: Start services ────────────────────────────────
start_services() {
    cd "$INSTALL_DIR"
    echo "[..] Building and starting all services..."
    docker compose -f docker-compose.prod.yml up -d --build

    echo ""
    echo "[..] Waiting for services to start..."
    sleep 10

    echo ""
    echo "  Service status:"
    docker compose -f docker-compose.prod.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
}

# ── Step 6: Post-deploy info ──────────────────────────────
show_info() {
    source "$INSTALL_DIR/.env" 2>/dev/null || true

    echo ""
    echo "========================================="
    echo "  Deployment Complete!"
    echo "========================================="
    echo ""

    if [ "$DOMAIN" != "trader.yourdomain.com" ] && [ -n "$DOMAIN" ]; then
        echo "  Dashboard: https://$DOMAIN"
    else
        IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_SERVER_IP")
        echo "  Dashboard: http://$IP"
    fi

    echo ""
    echo "  Useful commands:"
    echo "    cd $INSTALL_DIR"
    echo "    docker compose -f docker-compose.prod.yml logs -f        # All logs"
    echo "    docker compose -f docker-compose.prod.yml logs -f backend # Backend logs"
    echo "    docker compose -f docker-compose.prod.yml ps             # Service status"
    echo "    docker compose -f docker-compose.prod.yml restart        # Restart all"
    echo "    docker compose -f docker-compose.prod.yml down           # Stop all"
    echo ""
    echo "  IB Gateway VNC (for 2FA login):"
    echo "    Connect VNC viewer to YOUR_SERVER_IP:5900"
    echo "    Password: (VNC_PASSWORD from .env)"
    echo ""
    echo "  Next steps:"
    echo "    1. Connect to IB Gateway via VNC and complete 2FA if needed"
    echo "    2. Open the dashboard and verify broker connection"
    echo "    3. Check that paper trading mode is active"
    echo "    4. Monitor the Signal Feed for trading signals"
    echo ""
}

# ── Run ───────────────────────────────────────────────────
install_docker
clone_repo
configure_env
start_services
setup_ssl
show_info
