# Universal Knowledge Hub - Python 3.13.5 Setup Script (PowerShell)
# This script sets up the development environment for Python 3.13.5 on Windows

param(
    [switch]$SkipFrontend,
    [switch]$Force
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Universal Knowledge Hub for Python 3.13.5..." -ForegroundColor Green

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Python 3.13.5 is available
function Test-PythonVersion {
    Write-Status "Checking Python version..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.13") {
            Write-Success "Found Python: $pythonVersion"
            return $true
        } else {
            Write-Error "Python 3.13.5 is required but not found. Current version: $pythonVersion"
            Write-Status "Please install Python 3.13.5 from https://www.python.org/downloads/"
            return $false
        }
    } catch {
        Write-Error "Python is not installed or not in PATH"
        return $false
    }
}

# Create virtual environment
function New-VirtualEnvironment {
    Write-Status "Creating virtual environment..."
    
    if (Test-Path ".venv") {
        if ($Force) {
            Write-Warning "Virtual environment already exists. Removing old one..."
            Remove-Item -Recurse -Force ".venv"
        } else {
            Write-Warning "Virtual environment already exists. Use -Force to recreate."
            return
        }
    }
    
    try {
        python -m venv .venv
        Write-Success "Virtual environment created successfully"
    } catch {
        Write-Error "Failed to create virtual environment: $_"
        exit 1
    }
}

# Activate virtual environment
function Activate-VirtualEnvironment {
    Write-Status "Activating virtual environment..."
    
    try {
        & ".venv\Scripts\Activate.ps1"
        Write-Success "Virtual environment activated"
    } catch {
        Write-Error "Failed to activate virtual environment: $_"
        exit 1
    }
}

# Upgrade pip
function Update-Pip {
    Write-Status "Upgrading pip..."
    try {
        python -m pip install --upgrade pip
        Write-Success "Pip upgraded successfully"
    } catch {
        Write-Error "Failed to upgrade pip: $_"
        exit 1
    }
}

# Install dependencies
function Install-Dependencies {
    Write-Status "Installing core dependencies..."
    try {
        pip install -r requirements.txt
        
        Write-Status "Installing development dependencies..."
        pip install -r requirements-dev.txt
        
        Write-Status "Installing test dependencies..."
        pip install -r requirements-test.txt
        
        Write-Success "All dependencies installed successfully"
    } catch {
        Write-Error "Failed to install dependencies: $_"
        exit 1
    }
}

# Install pre-commit hooks
function Install-PreCommitHooks {
    Write-Status "Installing pre-commit hooks..."
    try {
        pre-commit install
        Write-Success "Pre-commit hooks installed"
    } catch {
        Write-Warning "Failed to install pre-commit hooks: $_"
        Write-Status "You can install pre-commit manually later"
    }
}

# Verify installation
function Test-Installation {
    Write-Status "Verifying installation..."
    
    try {
        # Check Python version in venv
        $venvPythonVersion = python --version 2>&1
        Write-Success "Virtual environment Python: $venvPythonVersion"
        
        # Check key packages
        Write-Status "Checking key packages..."
        python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
        python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')"
        python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
        python -c "import pytest; print(f'Pytest: {pytest.__version__}')"
        
        Write-Success "Installation verification completed"
    } catch {
        Write-Error "Installation verification failed: $_"
        exit 1
    }
}

# Create necessary directories
function New-Directories {
    Write-Status "Creating necessary directories..."
    
    $directories = @("data", "logs", "frontend\.next")
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Setup frontend
function Install-FrontendDependencies {
    if ($SkipFrontend) {
        Write-Warning "Frontend setup skipped due to -SkipFrontend flag"
        return
    }
    
    Write-Status "Setting up frontend dependencies..."
    
    try {
        # Check if npm is available
        $npmVersion = npm --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Set-Location frontend
            npm install
            Set-Location ..
            Write-Success "Frontend dependencies installed"
        } else {
            Write-Warning "Node.js/npm not found. Frontend setup skipped."
            Write-Status "Install Node.js from https://nodejs.org/ to set up frontend"
        }
    } catch {
        Write-Warning "Failed to install frontend dependencies: $_"
        Write-Status "You can install frontend dependencies manually later"
    }
}

# Main execution
function Main {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Universal Knowledge Hub Setup" -ForegroundColor Cyan
    Write-Host "Python 3.13.5 Development Environment" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    # Check Python version
    if (!(Test-PythonVersion)) {
        exit 1
    }
    
    # Setup steps
    New-VirtualEnvironment
    Activate-VirtualEnvironment
    Update-Pip
    Install-Dependencies
    Install-PreCommitHooks
    New-Directories
    Install-FrontendDependencies
    Test-Installation
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Success "Setup completed successfully!"
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Activate virtual environment:" -ForegroundColor White
    Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Configure environment variables:" -ForegroundColor White
    Write-Host "   Copy-Item env.template .env" -ForegroundColor Gray
    Write-Host "   # Edit .env with your API keys" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Start development servers:" -ForegroundColor White
    Write-Host "   # Backend: uvicorn api.main:app --reload" -ForegroundColor Gray
    Write-Host "   # Frontend: cd frontend && npm run dev" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Run tests:" -ForegroundColor White
    Write-Host "   pytest" -ForegroundColor Gray
    Write-Host ""
    Write-Host "For more information, see LOCAL_DEV_SETUP.md" -ForegroundColor White
    Write-Host "==========================================" -ForegroundColor Cyan
}

# Run main function
Main 