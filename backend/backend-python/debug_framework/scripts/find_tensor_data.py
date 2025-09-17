#!/usr/bin/env python3
"""
Tensor Data Location Finder

This script helps locate recorded tensor files and metadata in the debug framework.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

def find_tensor_data():
    """Find and display tensor data locations."""
    print("🔍 Tensor Data Location Guide")
    print("=" * 50)

    # Debug framework data directory
    data_dir = Path(__file__).parent.parent / "data"
    print(f"📁 Main Data Directory: {data_dir}")
    print(f"   Exists: {'✅' if data_dir.exists() else '❌'}")

    if data_dir.exists():
        print(f"   Contents: {list(data_dir.iterdir())}")

    # Expected structure
    expected_dirs = {
        'artifacts': 'Binary tensor files organized by session',
        'sessions': 'Session-specific metadata and organization',
        'exports': 'Exported session archives',
        'temp': 'Temporary files during processing'
    }

    print("\n📋 Expected Directory Structure:")
    for dirname, description in expected_dirs.items():
        dir_path = data_dir / dirname
        exists = "✅" if dir_path.exists() else "❌"
        print(f"   {exists} {dirname}/ - {description}")
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            if files:
                print(f"      Files: {len(files)} items")
                for f in files[:3]:  # Show first 3
                    print(f"        - {f.name}")
                if len(files) > 3:
                    print(f"        ... and {len(files) - 3} more")

    # Database files
    print("\n🗄️  Database Files:")
    db_files = [
        "debug_framework.db",
        "artifact_metadata.db"
    ]

    for db_file in db_files:
        db_path = data_dir / db_file
        exists = "✅" if db_path.exists() else "❌"
        size = f"({db_path.stat().st_size / 1024:.1f} KB)" if db_path.exists() else ""
        print(f"   {exists} {db_file} {size}")

    # Look for tensor files in temp directories
    print("\n🔍 Tensor Files in Temporary Locations:")
    temp_locations = ["/tmp", str(data_dir / "temp")]

    for temp_dir in temp_locations:
        if os.path.exists(temp_dir):
            tensor_files = []
            try:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.tensor') or 'tensor' in file.lower():
                            tensor_files.append(os.path.join(root, file))

                if tensor_files:
                    print(f"   📂 {temp_dir}: {len(tensor_files)} tensor files found")
                    for f in tensor_files[:5]:  # Show first 5
                        size = os.path.getsize(f) / 1024 if os.path.exists(f) else 0
                        print(f"      - {os.path.basename(f)} ({size:.1f} KB)")
                    if len(tensor_files) > 5:
                        print(f"      ... and {len(tensor_files) - 5} more")
                else:
                    print(f"   📂 {temp_dir}: No tensor files found")
            except Exception as e:
                print(f"   📂 {temp_dir}: Error scanning - {e}")

    # Instructions for accessing data
    print("\n📖 How to Access Your Tensor Data:")
    print("   1. Production tensor files: debug_framework/data/artifacts/session_*/")
    print("   2. Metadata database: debug_framework/data/artifact_metadata.db")
    print("   3. T063-T065 test tensors: /tmp/*.tensor (temporary)")
    print("   4. Session exports: debug_framework/data/exports/")

    print("\n🛠️  Python Code to Access Data:")
    print("""
   from debug_framework.services.artifact_manager import ArtifactManager

   # Initialize manager
   manager = ArtifactManager()

   # List all sessions
   sessions = manager.list_sessions()
   print(f"Found {len(sessions)} sessions")

   # Get artifacts for a session
   if sessions:
       session_id = sessions[0].session_id
       artifacts = manager.get_session_artifacts(session_id)
       print(f"Session {session_id} has {len(artifacts)} artifacts")
   """)

if __name__ == "__main__":
    find_tensor_data()