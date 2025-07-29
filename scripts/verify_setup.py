#!/usr/bin/env python3
"""
Universal Knowledge Hub - Setup Verification Script
Verifies that the Python 3.13.5 environment is properly configured.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_status(message: str) -> None:
    """Print a status message."""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def check_python_version() -> bool:
    """Check if Python 3.13.5 is available."""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor == 13:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.13.5 required, found {version.major}.{version.minor}.{version.micro}")
        return False

def check_virtual_environment() -> bool:
    """Check if running in a virtual environment."""
    print_status("Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Running in virtual environment")
        return True
    else:
        print_warning("Not running in virtual environment")
        return False

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    try:
        import_name = import_name or package_name
        importlib.import_module(import_name)
        print_success(f"✓ {package_name}")
        return True
    except ImportError:
        print_error(f"✗ {package_name} - not installed")
        return False

def get_package_version(package_name: str, import_name: str = None) -> Optional[str]:
    """Get the version of an installed package."""
    try:
        import_name = import_name or package_name
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            return module.__version__
        return "installed"
    except ImportError:
        return None

def check_required_packages() -> Dict[str, bool]:
    """Check all required packages."""
    print_status("Checking required packages...")
    
    packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
        'requests': 'requests',
        'python-dotenv': 'dotenv',
        'aiohttp': 'aiohttp',
        'numpy': 'numpy',
        'pytest': 'pytest',
        'black': 'black',
        'flake8': 'flake8',
        'mypy': 'mypy',
        'bandit': 'bandit',
    }
    
    results = {}
    for package, import_name in packages.items():
        results[package] = check_package(package, import_name)
    
    return results

def check_package_versions() -> None:
    """Check versions of key packages."""
    print_status("Checking package versions...")
    
    packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
        'requests': 'requests',
        'numpy': 'numpy',
        'pytest': 'pytest',
    }
    
    for package, import_name in packages.items():
        version = get_package_version(package, import_name)
        if version:
            print_success(f"{package}: {version}")

def check_directories() -> Dict[str, bool]:
    """Check if required directories exist."""
    print_status("Checking required directories...")
    
    directories = ['data', 'logs', 'frontend', 'api', 'agents', 'tests']
    results = {}
    
    for directory in directories:
        if Path(directory).exists():
            print_success(f"✓ {directory}/")
            results[directory] = True
        else:
            print_error(f"✗ {directory}/ - not found")
            results[directory] = False
    
    return results

def check_environment_files() -> Dict[str, bool]:
    """Check if environment configuration files exist."""
    print_status("Checking environment configuration...")
    
    files = ['.env', 'env.template']
    results = {}
    
    for file in files:
        if Path(file).exists():
            print_success(f"✓ {file}")
            results[file] = True
        else:
            print_warning(f"⚠ {file} - not found")
            results[file] = False
    
    return results

def check_nodejs() -> bool:
    """Check if Node.js is installed."""
    print_status("Checking Node.js installation...")
    
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Node.js: {version}")
            return True
        else:
            print_warning("Node.js not found")
            return False
    except FileNotFoundError:
        print_warning("Node.js not found")
        return False

def check_frontend_dependencies() -> bool:
    """Check if frontend dependencies are installed."""
    print_status("Checking frontend dependencies...")
    
    package_json = Path('frontend/package.json')
    node_modules = Path('frontend/node_modules')
    
    if package_json.exists() and node_modules.exists():
        print_success("✓ Frontend dependencies installed")
        return True
    else:
        print_warning("⚠ Frontend dependencies not installed")
        return False

def run_basic_tests() -> Dict[str, bool]:
    """Run basic import tests."""
    print_status("Running basic tests...")
    
    tests = {
        'Core imports': lambda: importlib.import_module('api.main'),
        'API structure': lambda: Path('api').exists(),
        'Agents structure': lambda: Path('agents').exists(),
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            test_func()
            print_success(f"✓ {test_name}")
            results[test_name] = True
        except Exception as e:
            print_error(f"✗ {test_name} - {e}")
            results[test_name] = False
    
    return results

def generate_report(
    python_ok: bool,
    venv_ok: bool,
    packages: Dict[str, bool],
    directories: Dict[str, bool],
    env_files: Dict[str, bool],
    nodejs_ok: bool,
    frontend_ok: bool,
    tests: Dict[str, bool]
) -> None:
    """Generate a comprehensive setup report."""
    
    print("\n" + "=" * 50)
    print("SETUP VERIFICATION REPORT")
    print("=" * 50)
    
    # Python Environment
    print("\n🐍 Python Environment:")
    print(f"  Python 3.13.5: {'✓' if python_ok else '✗'}")
    print(f"  Virtual Environment: {'✓' if venv_ok else '⚠'}")
    
    # Required Packages
    print("\n📦 Required Packages:")
    for package, installed in packages.items():
        status = "✓" if installed else "✗"
        print(f"  {package}: {status}")
    
    # Project Structure
    print("\n📁 Project Structure:")
    for directory, exists in directories.items():
        status = "✓" if exists else "✗"
        print(f"  {directory}/: {status}")
    
    # Configuration
    print("\n🔧 Configuration:")
    for file, exists in env_files.items():
        status = "✓" if exists else "⚠"
        print(f"  {file}: {status}")
    
    # Frontend
    print("\n🌐 Frontend:")
    print(f"  Node.js: {'✓' if nodejs_ok else '✗'}")
    print(f"  Dependencies: {'✓' if frontend_ok else '⚠'}")
    
    # Testing
    print("\n🧪 Testing:")
    for test_name, passed in tests.items():
        status = "✓" if passed else "✗"
        print(f"  {test_name}: {status}")
    
    # Overall Status
    print("\n📊 Overall Status:")
    all_packages_installed = all(packages.values())
    all_directories_exist = all(directories.values())
    all_tests_passed = all(tests.values())
    
    if python_ok and all_packages_installed and all_directories_exist and all_tests_passed:
        print_success("✅ Setup is complete and ready!")
    else:
        print_error("❌ Setup has critical issues:")
        if not python_ok:
            print_error("  - Python version issue")
        if not all_packages_installed:
            print_error("  - Missing required packages")
        if not all_directories_exist:
            print_error("  - Missing required directories")
        if not all_tests_passed:
            print_error("  - Basic tests failed")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not venv_ok:
        print("  - Activate virtual environment: source .venv/bin/activate")
    if not all_packages_installed:
        print("  - Install missing packages: pip install -r requirements.txt")
    if not frontend_ok:
        print("  - Install frontend dependencies: cd frontend && npm install")
    
    print("=" * 50)

def main() -> None:
    """Main verification function."""
    print("🔍 Universal Knowledge Hub - Setup Verification")
    print("=" * 50)
    
    # Run all checks
    python_ok = check_python_version()
    venv_ok = check_virtual_environment()
    packages = check_required_packages()
    check_package_versions()
    directories = check_directories()
    env_files = check_environment_files()
    nodejs_ok = check_nodejs()
    frontend_ok = check_frontend_dependencies()
    tests = run_basic_tests()
    
    # Generate report
    generate_report(
        python_ok, venv_ok, packages, directories, 
        env_files, nodejs_ok, frontend_ok, tests
    )
    
    # Exit with appropriate code
    all_packages_installed = all(packages.values())
    all_directories_exist = all(directories.values())
    all_tests_passed = all(tests.values())
    
    if python_ok and all_packages_installed and all_directories_exist and all_tests_passed:
        sys.exit(0)
    else:
        print_error("Setup verification failed. Issues: Required packages")
        sys.exit(1)

if __name__ == "__main__":
    main() 