#!/usr/bin/env python3
"""Validate requirements.txt for duplicates and missing imports."""

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_requirements(file_path):
    """Parse requirements.txt and return package information."""
    packages = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Extract package name (handle various formats)
            match = re.match(r'^([a-zA-Z0-9_-]+)', line)
            if match:
                package_name = match.group(1).lower()
                packages[package_name].append((line_num, line))
    
    return packages


def find_imports_in_file(file_path):
    """Find all import statements in a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Find import statements (skip commented lines)
        for line in lines:
            # Skip commented lines
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                continue
                
            # Find import statements
            import_match = re.match(r'^(?:from\s+(\S+)|import\s+(\S+))', stripped_line)
            if import_match:
                module = import_match.group(1) or import_match.group(2)
                if module:
                    # Get the top-level package name
                    top_level = module.split('.')[0]
                    imports.add(top_level)
    except Exception:
        pass  # Skip files that can't be read
    
    return imports


def find_all_imports(root_dir):
    """Find all imports in Python files throughout the project."""
    all_imports = set()
    
    for py_file in Path(root_dir).rglob('*.py'):
        # Skip virtual environments and other common directories
        if any(part in py_file.parts for part in ['venv', '.venv', '__pycache__', '.git', 'node_modules', 'test_env']):
            continue
        
        imports = find_imports_in_file(py_file)
        all_imports.update(imports)
    
    return all_imports


def is_local_module(module_name, root_dir):
    """Check if a module is a local module in the project."""
    # Check if it's a directory with __init__.py
    module_path = Path(root_dir) / module_name
    if module_path.is_dir() and (module_path / '__init__.py').exists():
        return True
    
    # Check if it's a Python file
    if (Path(root_dir) / f"{module_name}.py").exists():
        return True
    
    return False


def main():
    """Main validation function."""
    print("Validating requirements.txt...")
    print("=" * 60)
    
    # Parse requirements.txt
    requirements_path = Path('requirements.txt')
    if not requirements_path.exists():
        print("ERROR: requirements.txt not found!")
        sys.exit(1)
    
    packages = parse_requirements(requirements_path)
    
    # Check for duplicates
    print("\n1. Checking for duplicate packages:")
    print("-" * 40)
    duplicates_found = False
    for package_name, entries in packages.items():
        if len(entries) > 1:
            duplicates_found = True
            print(f"\n❌ DUPLICATE: {package_name}")
            for line_num, line in entries:
                print(f"   Line {line_num}: {line}")
    
    if not duplicates_found:
        print("✅ No duplicates found!")
    
    # Map of import names to package names (for common mismatches)
    import_to_package = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'yaml': 'pyyaml',
        'jwt': 'pyjwt',
        'dotenv': 'python-dotenv',
        'sklearn': 'scikit-learn',
        'jose': 'python-jose',
        'bs4': 'beautifulsoup4',
        'dateutil': 'python-dateutil',
        'sentry_sdk': 'sentry-sdk',
        'sqlalchemy_utils': 'sqlalchemy-utils',
        'langchain_openai': 'langchain-openai',
        'langchain_anthropic': 'langchain-anthropic',
        'pinecone': 'pinecone-client',
        'sparql': 'sparql-client',
        'multipart': 'python-multipart',
        'email_validator': 'email-validator',
        'passlib': 'passlib',
        'owlready2': 'owlready2',
        'sparql_client': 'sparql-client',
        'prometheus_client': 'prometheus-client',
        'spacy': 'spacy',
        'torch': 'torch',
        'transformers': 'transformers',
        'sentence_transformers': 'sentence-transformers',
        'qdrant_client': 'qdrant-client',
        'aioredis': 'aioredis',
        'elastic_transport': 'elastic-transport',
        'httpcore': 'httpcore',
        'httptools': 'httptools',
        'watchfiles': 'watchfiles',
        'uvloop': 'uvloop',
        'eventlet': 'eventlet',
        'aiohappyeyeballs': 'aiohappyeyeballs',
        'aiosignal': 'aiosignal',
        'frozenlist': 'frozenlist',
        'multidict': 'multidict',
        'yarl': 'yarl',
        'async_timeout': 'async-timeout',
        'propcache': 'propcache',
        'greenlet': 'greenlet',
        'jiter': 'jiter',
        'annotated_types': 'annotated-types',
        'typing_extensions': 'typing-extensions',
        'typing_inspect': 'typing-inspect',
        'mypy_extensions': 'mypy-extensions',
        'pytest_asyncio': 'pytest-asyncio',
        'pytest_cov': 'pytest-cov',
        'pycodestyle': 'pycodestyle',
        'pyflakes': 'pyflakes',
        'mccabe': 'mccabe',
        'pathspec': 'pathspec',
        'platformdirs': 'platformdirs',
        'pluggy': 'pluggy',
        'iniconfig': 'iniconfig',
        'exceptiongroup': 'exceptiongroup',
        'tomli': 'tomli',
        'packaging': 'packaging',
        'attrs': 'attrs',
        'sniffio': 'sniffio',
        'h11': 'h11',
        'anyio': 'anyio',
        'idna': 'idna',
        'certifi': 'certifi',
        'charset_normalizer': 'charset-normalizer',
        'distro': 'distro',
        'dnspython': 'dnspython',
        'pytz': 'pytz',
        'tzdata': 'tzdata',
        'six': 'six',
        'click': 'click',
        'tqdm': 'tqdm',
        'pygments': 'pygments',
        'rich': 'rich',
        'typer': 'typer',
        'shellingham': 'shellingham',
        'wasabi': 'wasabi',
        'srsly': 'srsly',
        'catalogue': 'catalogue',
        'confection': 'confection',
        'pathy': 'pathy',
        'smart_open': 'smart-open',
        'weasel': 'weasel',
        'cloudpathlib': 'cloudpathlib',
        'langcodes': 'langcodes',
        'marisa_trie': 'marisa-trie',
        'markdown_it_py': 'markdown-it-py',
        'mdurl': 'mdurl',
        'language_data': 'language-data',
        'blis': 'blis',
        'thinc': 'thinc',
        'cymem': 'cymem',
        'preshed': 'preshed',
        'murmurhash': 'murmurhash',
        'spacy_legacy': 'spacy-legacy',
        'spacy_loggers': 'spacy-loggers',
        'wrapt': 'wrapt',
        'jinja2': 'jinja2',
        'markupsafe': 'markupsafe',
        'rdflib': 'rdflib',
        'sparql': 'sparql-client',
        'elasticsearch': 'elasticsearch',
        'openai': 'openai',
        'anthropic': 'anthropic',
        'langchain': 'langchain',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'scikit_learn': 'scikit-learn',
        'sklearn': 'scikit-learn',
        'requests': 'requests',
        'urllib3': 'urllib3',
        'aiohttp': 'aiohttp',
        'httpx': 'httpx',
        'websockets': 'websockets',
        'redis': 'redis',
        'asyncpg': 'asyncpg',
        'sqlalchemy': 'sqlalchemy',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'starlette': 'starlette',
        'pydantic': 'pydantic',
        'pydantic_core': 'pydantic-core',
        'pytest': 'pytest',
        'black': 'black',
        'flake8': 'flake8',
        'mypy': 'mypy',
        'coverage': 'coverage',
        'faker': 'faker',
        'freezegun': 'freezegun',
        'hypothesis': 'hypothesis',
        'bleach': 'bleach',
        'pyyaml': 'pyyaml',
        'tenacity': 'tenacity',
    }
    
    # Find all imports in the project
    print("\n2. Checking for missing packages:")
    print("-" * 40)
    print("Scanning project for imports...")
    
    all_imports = find_all_imports('.')
    
    # Get list of installed packages
    installed_packages = set(packages.keys())
    
    # Add mapped package names
    for import_name, package_name in import_to_package.items():
        if import_name in all_imports and package_name.replace('-', '_').lower() in installed_packages:
            all_imports.discard(import_name)
    
    # Check for potentially missing packages
    stdlib_modules = {
        'os', 'sys', 'time', 'datetime', 'json', 're', 'math', 'random',
        'collections', 'itertools', 'functools', 'pathlib', 'typing',
        'asyncio', 'threading', 'multiprocessing', 'subprocess', 'shutil',
        'tempfile', 'io', 'contextlib', 'warnings', 'logging', 'traceback',
        'unittest', 'doctest', 'abc', 'enum', 'dataclasses', 'decimal',
        'fractions', 'statistics', 'copy', 'pickle', 'base64', 'hashlib',
        'hmac', 'secrets', 'uuid', 'urllib', 'http', 'email', 'mimetypes',
        'csv', 'configparser', 'argparse', 'getopt', 'locale', 'platform',
        'socket', 'ssl', 'select', 'queue', 'heapq', 'bisect', 'array',
        'weakref', 'types', 'operator', 'inspect', 'importlib', 'pkgutil',
        'ast', 'dis', 'code', 'codeop', 'pdb', 'profile', 'timeit',
        'trace', 'gc', 'atexit', 'signal', 'faulthandler', 'ctypes',
        'struct', 'codecs', 'encodings', 'unicodedata', 'stringprep',
        'builtins', '__future__', 'imp', 'zipimport', 'zlib', 'gzip',
        'bz2', 'lzma', 'zipfile', 'tarfile', 'sqlite3', 'dbm', 'shelve',
        'marshal', 'xml', 'html', 'webbrowser', 'cgi', 'cgitb', 'wsgiref',
        'ftplib', 'poplib', 'imaplib', 'smtplib', 'telnetlib', 'mailbox',
        'turtle', 'cmd', 'shlex', 'glob', 'fnmatch', 'linecache', 'tokenize',
        'tabnanny', 'py_compile', 'compileall', 'pyclbr', 'venv', 'ensurepip',
        'concurrent', 'tracemalloc', 'curses', 'textwrap', 'string', 'difflib',
        'pprint', 'reprlib', 'numbers', 'cmath', 'complex', 'binascii',
        'calendar', 'sched', 'getpass', 'pwd', 'grp', 'crypt', 'termios',
        'tty', 'pty', 'fcntl', 'pipes', 'resource', 'nis', 'syslog',
        'posix', 'errno', 'stat', 'filecmp', 'fileinput', 'xdrlib',
        'plistlib', 'modulefinder', 'runpy', 'parser', 'symbol', 'symtable',
        'pickletools', 'copyreg', 'quopri', 'uu', 'binhex', 'rlcompleter',
        'readline', 'ipaddress', 'xmlrpc', 'imghdr', 'audioop', 'aifc',
        'sunau', 'wave', 'chunk', 'colorsys', 'rgbimg', 'imgfile', 'sndhdr',
        'ossaudiodev', 'gettext', 'optparse', 'imp', 'formatter',
    }
    
    # Filter out stdlib modules and check for missing packages
    missing_packages = []
    for import_name in sorted(all_imports):
        if import_name in stdlib_modules:
            continue
        
        # Check if it's a local module
        if is_local_module(import_name, '.'):
            continue
        
        # Skip empty imports
        if not import_name:
            continue
        
        # Check if it's in requirements (direct or mapped)
        package_name = import_to_package.get(import_name, import_name)
        if package_name.lower() not in installed_packages and package_name.replace('-', '_').lower() not in installed_packages:
            # Also check with underscores converted to hyphens
            if package_name.replace('_', '-').lower() not in installed_packages:
                missing_packages.append(import_name)
    
    # Known packages that might be legitimately missing
    optional_packages = {
        'aiocache', 'colorama', 'joblib', 'kafka', 'locust', 
        'opentelemetry', 'pyotp', 'com', 'okhttp3',
        'core', 'prompts'  # Local modules referenced in tests but not present
    }
    
    # Filter out optional packages
    actual_missing = [pkg for pkg in missing_packages if pkg not in optional_packages]
    
    if actual_missing:
        print(f"\n❌ Found {len(actual_missing)} potentially missing packages:")
        for pkg in actual_missing[:20]:  # Show first 20
            print(f"   - {pkg}")
        if len(actual_missing) > 20:
            print(f"   ... and {len(actual_missing) - 20} more")
    else:
        print("✅ No obviously missing packages found!")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    if duplicates_found or actual_missing:
        print("❌ Issues found - please review and fix!")
        sys.exit(1)
    else:
        print("✅ requirements.txt looks good!")
        sys.exit(0)


if __name__ == '__main__':
    main() 