#!/bin/bash
# =============================================================================
# Deployment Script for Quantum Dengue STPP
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="quantum-dengue-stpp"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
USERNAME="${DOCKER_USERNAME:-}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { log_error "Docker Compose is required but not installed."; exit 1; }
    
    log_info "All requirements met."
}

build_images() {
    log_info "Building Docker images..."
    
    docker-compose build --parallel
    
    log_info "Build complete."
}

run_tests() {
    log_info "Running tests..."
    
    docker-compose run --rm test
    
    log_info "Tests complete."
}

start_services() {
    log_info "Starting services..."
    
    docker-compose up -d
    
    log_info "Services started."
    log_info "API available at: http://localhost:8000"
    log_info "API docs at: http://localhost:8000/docs"
    log_info "Grafana available at: http://localhost:3000"
}

stop_services() {
    log_info "Stopping services..."
    
    docker-compose down
    
    log_info "Services stopped."
}

deploy_staging() {
    log_info "Deploying to staging..."
    
    check_requirements
    build_images
    
    docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
    
    log_info "Staging deployment complete."
}

deploy_production() {
    log_info "Deploying to production..."
    
    if [ -z "$USERNAME" ]; then
        log_error "DOCKER_USERNAME not set. Cannot deploy to production."
        exit 1
    fi
    
    check_requirements
    
    # Build and push to registry
    log_info "Building and tagging images..."
    docker-compose build
    docker tag ${PROJECT_NAME}_api ${USERNAME}/${PROJECT_NAME}:latest
    docker tag ${PROJECT_NAME}_api ${USERNAME}/${PROJECT_NAME}:$(git rev-parse --short HEAD)
    
    log_info "Pushing images to registry..."
    docker push ${USERNAME}/${PROJECT_NAME}:latest
    docker push ${USERNAME}/${PROJECT_NAME}:$(git rev-parse --short HEAD)
    
    log_info "Production deployment complete."
}

backup_data() {
    log_info "Backing up data..."
    
    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    docker-compose exec -T redis redis-cli BGSAVE
    sleep 2
    
    docker-compose exec -T redis tar czf /tmp/redis_backup.tar.gz /data
    docker-compose cp redis:/tmp/redis_backup.tar.gz "$BACKUP_DIR/"
    
    tar czf "$BACKUP_DIR/app_data.tar.gz" ./data ./outputs 2>/dev/null || true
    
    log_info "Backup saved to: $BACKUP_DIR"
}

restore_backup() {
    BACKUP_DIR="$1"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_DIR"
        exit 1
    fi
    
    log_info "Restoring from backup: $BACKUP_DIR"
    
    docker-compose exec -T redis tar xzf /tmp/redis_backup.tar.gz -C /data || true
    tar xzf "$BACKUP_DIR/app_data.tar.gz" 2>/dev/null || true
    
    log_info "Restore complete."
}

show_status() {
    log_info "Service Status:"
    docker-compose ps
}

show_logs() {
    SERVICE="${1:-api}"
    log_info "Logs for $SERVICE:"
    docker-compose logs -f --tail=100 "$SERVICE"
}

# Main
case "${1:-}" in
    build)
        build_images
        ;;
    test)
        run_tests
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    deploy-staging)
        deploy_staging
        ;;
    deploy-prod)
        deploy_production
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_backup "${2:-}"
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-api}"
        ;;
    *)
        echo "Usage: $0 {build|test|start|stop|restart|deploy-staging|deploy-prod|backup|restore|status|logs}"
        echo ""
        echo "Commands:"
        echo "  build         Build Docker images"
        echo "  test          Run tests"
        echo "  start         Start all services"
        echo "  stop          Stop all services"
        echo "  restart       Restart all services"
        echo "  deploy-staging  Deploy to staging"
        echo "  deploy-prod   Deploy to production"
        echo "  backup        Backup data"
        echo "  restore <dir> Restore from backup"
        echo "  status        Show service status"
        echo "  logs [svc]    Show logs for service"
        exit 1
        ;;
esac
