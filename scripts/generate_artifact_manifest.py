#!/usr/bin/env python3

"""
Artifact Manifest Generator - Auto-generate expected file patterns for validation

This script analyzes existing artifacts to create a manifest of expected files
for each operation and case, enabling robust validation during transfer.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def analyze_artifacts_directory(artifacts_dir: Path) -> Dict:
    """Analyze artifacts directory and generate expected file patterns."""
    
    manifest = {
        "version": "1.0",
        "generated_from": str(artifacts_dir),
        "operations": {},
        "validation_rules": {
            "required_files": ["meta.json"],
            "file_extensions": [".bin", ".json"],
            "max_missing_files": 0
        }
    }
    
    operation_patterns = defaultdict(lambda: defaultdict(set))
    
    # Walk through all artifacts
    for op_dir in artifacts_dir.iterdir():
        if not op_dir.is_dir():
            continue
            
        op_name = op_dir.name
        manifest["operations"][op_name] = {"cases": {}, "file_patterns": set()}
        
        for case_dir in op_dir.iterdir():
            if not case_dir.is_dir():
                continue
                
            case_id = case_dir.name
            case_files = []
            
            # Collect all files in this case
            for file_path in case_dir.iterdir():
                if file_path.is_file():
                    case_files.append(file_path.name)
                    operation_patterns[op_name]["all_files"].add(file_path.name)
                    
            # Read meta.json to understand the operation better
            meta_path = case_dir / "meta.json"
            meta_data = {}
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not read meta.json for {op_name}/{case_id}: {e}")
            
            manifest["operations"][op_name]["cases"][case_id] = {
                "files": sorted(case_files),
                "file_count": len(case_files),
                "meta": meta_data
            }
    
    # Generate file patterns for each operation
    for op_name, data in operation_patterns.items():
        all_files = sorted(data["all_files"])
        manifest["operations"][op_name]["file_patterns"] = all_files
        
        # Analyze common patterns
        file_types = defaultdict(int)
        for filename in all_files:
            if filename.endswith('.bin'):
                file_types['binary'] += 1
            elif filename.endswith('.json'):
                file_types['metadata'] += 1
                
        manifest["operations"][op_name]["statistics"] = {
            "unique_filenames": len(all_files),
            "file_types": dict(file_types)
        }
    
    return manifest

def validate_artifacts_against_manifest(artifacts_dir: Path, manifest: Dict) -> Tuple[bool, List[str]]:
    """Validate current artifacts against expected manifest."""
    
    issues = []
    all_valid = True
    
    # Check each operation
    for op_name, op_data in manifest["operations"].items():
        op_path = artifacts_dir / op_name
        
        if not op_path.exists():
            issues.append(f"Missing operation directory: {op_name}")
            all_valid = False
            continue
            
        # Check each case
        for case_id, case_data in op_data["cases"].items():
            case_path = op_path / case_id
            
            if not case_path.exists():
                issues.append(f"Missing case directory: {op_name}/{case_id}")
                all_valid = False
                continue
                
            # Check expected files
            expected_files = set(case_data["files"])
            actual_files = set(f.name for f in case_path.iterdir() if f.is_file())
            
            missing_files = expected_files - actual_files
            extra_files = actual_files - expected_files
            
            if missing_files:
                issues.append(f"Missing files in {op_name}/{case_id}: {sorted(missing_files)}")
                all_valid = False
                
            if extra_files:
                issues.append(f"Extra files in {op_name}/{case_id}: {sorted(extra_files)}")
                
            # Validate meta.json structure
            meta_path = case_path / "meta.json"
            if meta_path.exists() and "meta" in case_data:
                try:
                    with open(meta_path, 'r') as f:
                        current_meta = json.load(f)
                    
                    expected_meta = case_data["meta"]
                    
                    # Check critical fields
                    for key in ["op", "case_id"]:
                        if key in expected_meta:
                            if current_meta.get(key) != expected_meta.get(key):
                                issues.append(f"Meta mismatch in {op_name}/{case_id}: {key} = {current_meta.get(key)} vs {expected_meta.get(key)}")
                                
                except Exception as e:
                    issues.append(f"Could not validate meta.json for {op_name}/{case_id}: {e}")
    
    return all_valid, issues

def generate_expected_files_summary(manifest: Dict) -> Dict:
    """Generate a summary of expected files for quick validation."""
    
    summary = {
        "total_operations": len(manifest["operations"]),
        "total_cases": 0,
        "total_files": 0,
        "operations_summary": {}
    }
    
    for op_name, op_data in manifest["operations"].items():
        case_count = len(op_data["cases"])
        file_count = sum(len(case_data["files"]) for case_data in op_data["cases"].values())
        
        summary["total_cases"] += case_count
        summary["total_files"] += file_count
        
        summary["operations_summary"][op_name] = {
            "cases": case_count,
            "files": file_count,
            "pattern_files": len(op_data["file_patterns"])
        }
    
    return summary

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_artifact_manifest.py <command> [args]")
        print("Commands:")
        print("  generate <artifacts_dir>     - Generate manifest from existing artifacts")
        print("  validate <artifacts_dir> <manifest_file> - Validate artifacts against manifest")
        print("  summary <manifest_file>      - Show summary of expected files")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate":
        if len(sys.argv) != 3:
            print("Usage: python3 generate_artifact_manifest.py generate <artifacts_dir>")
            sys.exit(1)
            
        artifacts_dir = Path(sys.argv[2])
        
        if not artifacts_dir.exists():
            print(f"Error: Artifacts directory not found: {artifacts_dir}")
            sys.exit(1)
            
        print(f"🔍 Analyzing artifacts in: {artifacts_dir}")
        manifest = analyze_artifacts_directory(artifacts_dir)
        
        # Save manifest
        manifest_file = artifacts_dir.parent / "artifact_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"✅ Generated manifest: {manifest_file}")
        
        # Show summary
        summary = generate_expected_files_summary(manifest)
        print(f"📊 Summary: {summary['total_operations']} operations, {summary['total_cases']} cases, {summary['total_files']} files")
        
        for op_name, op_summary in summary["operations_summary"].items():
            print(f"  • {op_name}: {op_summary['cases']} cases, {op_summary['files']} files")
    
    elif command == "validate":
        if len(sys.argv) != 4:
            print("Usage: python3 generate_artifact_manifest.py validate <artifacts_dir> <manifest_file>")
            sys.exit(1)
            
        artifacts_dir = Path(sys.argv[2])
        manifest_file = Path(sys.argv[3])
        
        if not artifacts_dir.exists():
            print(f"Error: Artifacts directory not found: {artifacts_dir}")
            sys.exit(1)
            
        if not manifest_file.exists():
            print(f"Error: Manifest file not found: {manifest_file}")
            sys.exit(1)
            
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        print(f"🔍 Validating artifacts against manifest...")
        
        is_valid, issues = validate_artifacts_against_manifest(artifacts_dir, manifest)
        
        if is_valid:
            print("✅ All artifacts are valid!")
        else:
            print("❌ Validation failed:")
            for issue in issues:
                print(f"  • {issue}")
            sys.exit(1)
    
    elif command == "summary":
        if len(sys.argv) != 3:
            print("Usage: python3 generate_artifact_manifest.py summary <manifest_file>")
            sys.exit(1)
            
        manifest_file = Path(sys.argv[2])
        
        if not manifest_file.exists():
            print(f"Error: Manifest file not found: {manifest_file}")
            sys.exit(1)
            
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        summary = generate_expected_files_summary(manifest)
        
        print("📊 Artifact Manifest Summary")
        print(f"Total: {summary['total_operations']} operations, {summary['total_cases']} cases, {summary['total_files']} files")
        print("\nOperation Details:")
        
        for op_name, op_summary in summary["operations_summary"].items():
            print(f"  • {op_name:<25} {op_summary['cases']:>3} cases, {op_summary['files']:>3} files")
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Valid commands: generate, validate, summary")
        sys.exit(1)

if __name__ == "__main__":
    main()