#!/bin/bash

# 🧹 Universal Knowledge Platform - Project Cleanup Script
# Removes duplicate and unwanted files/folders

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to safely remove files/directories
safe_remove() {
    local target="$1"
    local description="$2"
    
    if [[ -e "$target" ]]; then
        log_info "Removing $description: $target"
        rm -rf "$target"
        log_success "Removed: $target"
    else
        log_warning "Not found: $target"
    fi
}

# Function to merge directories
merge_directories() {
    local source="$1"
    local destination="$2"
    local description="$3"
    
    if [[ -d "$source" && -d "$destination" ]]; then
        log_info "Merging $description: $source -> $destination"
        cp -r "$source"/* "$destination"/ 2>/dev/null || true
        rm -rf "$source"
        log_success "Merged: $source into $destination"
    elif [[ -d "$source" ]]; then
        log_info "Moving $description: $source -> $destination"
        mv "$source" "$destination"
        log_success "Moved: $source to $destination"
    fi
}

# Main cleanup function
main() {
    log_info "Starting project cleanup..."
    
    # Remove duplicate infrastructure directories
    log_info "Cleaning up duplicate infrastructure directories..."
    
    # Merge infra/ into infrastructure/
    merge_directories "infra" "infrastructure" "infrastructure configuration"
    
    # Remove old k8s directory (replaced by infrastructure/kubernetes)
    safe_remove "k8s" "old Kubernetes manifests"
    
    # Remove old backend directory (functionality moved to services/)
    safe_remove "backend" "old backend directory"
    
    # Remove old core directory (functionality moved to services/)
    safe_remove "core" "old core directory"
    
    # Merge docs/ into documentation/
    merge_directories "docs" "documentation" "documentation"
    
    # Remove old monitoring directory (moved to infrastructure/monitoring)
    safe_remove "monitoring" "old monitoring directory"
    
    # Remove duplicate documentation files
    log_info "Cleaning up duplicate documentation files..."
    
    # Remove old documentation files that are now in documentation/
    safe_remove "docs/ENTERPRISE_DEPLOYMENT_GUIDE.md" "duplicate deployment guide"
    
    # Remove old progress and test reports (keep only the latest)
    safe_remove "CLEANUP_SUMMARY.md" "old cleanup summary"
    safe_remove "COMPREHENSIVE_TEST_REPORT.md" "old test report"
    safe_remove "COMPLETE_30_DAY_PROGRESS.md" "old progress report"
    
    # Remove temporary files
    log_info "Cleaning up temporary files..."
    safe_remove "api.log" "API log file"
    safe_remove ".pytest_cache" "pytest cache"
    
    # Remove old architecture files (replaced by ENTERPRISE_ARCHITECTURE.md)
    safe_remove "architecture" "old architecture directory"
    
    # Remove old prompts directory (moved to llmops/prompt-engineering)
    safe_remove "prompts" "old prompts directory"
    
    # Remove old .venv directory (should be recreated)
    safe_remove ".venv" "old virtual environment"
    
    # Clean up empty directories
    log_info "Removing empty directories..."
    find . -type d -empty -delete 2>/dev/null || true
    
    # Remove __pycache__ directories
    log_info "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove .DS_Store files (macOS)
    log_info "Removing macOS system files..."
    find . -name ".DS_Store" -delete 2>/dev/null || true
    
    # Remove temporary files
    log_info "Removing temporary files..."
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name "*.temp" -delete 2>/dev/null || true
    find . -name "*~" -delete 2>/dev/null || true
    
    # Clean up git
    log_info "Cleaning up git..."
    git gc --aggressive --prune=now 2>/dev/null || true
    
    log_success "Project cleanup completed!"
    
    # Show final structure
    log_info "Final project structure:"
    echo
    tree -L 2 -I 'node_modules|__pycache__|.git|.venv' 2>/dev/null || find . -maxdepth 2 -type d | head -20
}

# Run cleanup
main "$@" 