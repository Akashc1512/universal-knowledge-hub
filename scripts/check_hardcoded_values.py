#!/usr/bin/env python3
"""
🔍 HARDCODED VALUES CHECKER
Universal Knowledge Platform - Environment Variable Audit

This script identifies hardcoded values in the codebase that should be replaced
with environment variables for better configuration management.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_hardcoded_values(file_path: str) -> List[Dict[str, Any]]:
    """Find hardcoded values in a file."""
    hardcoded_values = []
    
    # Patterns to look for
    patterns = [
        # URLs and endpoints
        (r'http://localhost:\d+', 'URL', 'Database/Service URL'),
        (r'https://localhost:\d+', 'URL', 'Database/Service URL'),
        (r'bolt://localhost:\d+', 'URL', 'Neo4j connection URL'),
        (r'redis://localhost:\d+', 'URL', 'Redis connection URL'),
        (r'postgresql://.*@localhost:\d+', 'URL', 'PostgreSQL connection URL'),
        
        # Port numbers
        (r':8002', 'PORT', 'API port'),
        (r':8003', 'PORT', 'Test API port'),
        (r':8000', 'PORT', 'Development port'),
        (r':6333', 'PORT', 'Vector DB port'),
        (r':9200', 'PORT', 'Elasticsearch port'),
        (r':7687', 'PORT', 'Neo4j port'),
        (r':8890', 'PORT', 'SPARQL endpoint port'),
        (r':7200', 'PORT', 'Knowledge Graph port'),
        (r':6379', 'PORT', 'Redis port'),
        (r':5432', 'PORT', 'PostgreSQL port'),
        
        # Timeouts and intervals
        (r'timeout=30', 'TIMEOUT', 'Request timeout'),
        (r'timeout=10', 'TIMEOUT', 'Short timeout'),
        (r'timeout=5', 'TIMEOUT', 'Quick timeout'),
        (r'sleep\(3\)', 'INTERVAL', 'Sleep interval'),
        (r'sleep\(2\)', 'INTERVAL', 'Short sleep'),
        (r'ttl=3600', 'TTL', 'Cache TTL'),
        (r'ttl=7200', 'TTL', 'Long cache TTL'),
        (r'ttl=1800', 'TTL', 'Short cache TTL'),
        
        # Limits and thresholds
        (r'max_workers=5', 'LIMIT', 'Concurrent workers'),
        (r'max_size=1000', 'LIMIT', 'Cache size'),
        (r'max_size=10000', 'LIMIT', 'Large cache size'),
        (r'limit=10', 'LIMIT', 'Result limit'),
        (r'limit=20', 'LIMIT', 'Large result limit'),
        (r'limit=50', 'LIMIT', 'Very large result limit'),
        
        # Token budgets
        (r'1000000', 'BUDGET', 'Daily token budget'),
        (r'10000', 'BUDGET', 'Max tokens per query'),
        (r'1000', 'BUDGET', 'Default token budget'),
        
        # Confidence thresholds
        (r'0\.92', 'THRESHOLD', 'Similarity threshold'),
        (r'0\.95', 'THRESHOLD', 'High similarity threshold'),
        (r'0\.7', 'THRESHOLD', 'Confidence threshold'),
        
        # Hostnames
        (r'localhost', 'HOST', 'Local hostname'),
        (r'127\.0\.0\.1', 'HOST', 'Local IP'),
        (r'0\.0\.0\.0', 'HOST', 'Bind address'),
        
        # Database names
        (r'knowledge_base', 'DB', 'Database name'),
        (r'ukp_db', 'DB', 'Application database'),
        (r'universal-knowledge-hub', 'DB', 'Index name'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern, category, description in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        hardcoded_values.append({
                            'file': file_path,
                            'line': line_num,
                            'line_content': line.strip(),
                            'value': match.group(),
                            'category': category,
                            'description': description,
                            'suggestion': f'Replace with environment variable for {description.lower()}'
                        })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return hardcoded_values

def scan_directory(directory: str) -> List[Dict[str, Any]]:
    """Scan directory for Python files and find hardcoded values."""
    all_hardcoded = []
    
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and git directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '.venv']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                hardcoded = find_hardcoded_values(file_path)
                all_hardcoded.extend(hardcoded)
    
    return all_hardcoded

def generate_report(hardcoded_values: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive report of hardcoded values."""
    if not hardcoded_values:
        return "✅ No hardcoded values found!"
    
    # Group by category
    by_category = {}
    for item in hardcoded_values:
        category = item['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(item)
    
    report = []
    report.append("🔍 HARDCODED VALUES AUDIT REPORT")
    report.append("=" * 60)
    report.append(f"Total hardcoded values found: {len(hardcoded_values)}")
    report.append("")
    
    for category, items in by_category.items():
        report.append(f"📁 {category.upper()} VALUES ({len(items)} found)")
        report.append("-" * 40)
        
        # Group by file
        by_file = {}
        for item in items:
            file = item['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(item)
        
        for file, file_items in by_file.items():
            report.append(f"📄 {file}")
            for item in file_items:
                report.append(f"   Line {item['line']}: {item['value']}")
                report.append(f"   → {item['suggestion']}")
                report.append("")
    
    # Summary of suggested environment variables
    report.append("🔧 SUGGESTED ENVIRONMENT VARIABLES")
    report.append("=" * 60)
    
    suggested_vars = set()
    for item in hardcoded_values:
        category = item['category']
        if category == 'URL':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_URL")
        elif category == 'PORT':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_PORT")
        elif category == 'HOST':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_HOST")
        elif category == 'TIMEOUT':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_TIMEOUT")
        elif category == 'TTL':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_TTL")
        elif category == 'LIMIT':
            suggested_vars.add(f"MAX_{item['description'].upper().replace(' ', '_')}")
        elif category == 'BUDGET':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}")
        elif category == 'THRESHOLD':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_THRESHOLD")
        elif category == 'DB':
            suggested_vars.add(f"{item['description'].upper().replace(' ', '_')}_NAME")
    
    for var in sorted(suggested_vars):
        report.append(f"   {var}")
    
    return "\n".join(report)

def main():
    """Main function."""
    print("🔍 Scanning for hardcoded values...")
    
    # Scan the project directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hardcoded_values = scan_directory(project_root)
    
    # Generate and print report
    report = generate_report(hardcoded_values)
    print(report)
    
    # Save report to file
    report_file = os.path.join(project_root, 'HARDCODED_VALUES_REPORT.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    if hardcoded_values:
        print(f"\n⚠️  Found {len(hardcoded_values)} hardcoded values that should be replaced with environment variables.")
        print("💡 Use the env.template file to add the suggested environment variables to your .env file.")
    else:
        print("\n✅ No hardcoded values found! Your codebase is properly configured.")

if __name__ == "__main__":
    main() 