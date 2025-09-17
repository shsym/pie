#!/usr/bin/env python3
"""
CLI Tensor Recording Auto-Detection

Command-line utility for automatically detecting tensor recordings from
L4MA model inference and organizing them through the artifact management system.

This utility integrates with the proven T063-T065 tensor validation system
to automatically discover, validate, and organize tensor recordings.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile

# Add backend-python to path
backend_python_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_python_path))

try:
    from debug_framework.services.artifact_manager import ArtifactManager, ArtifactType
    from debug_framework.models.tensor_recording import TensorRecording
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Artifact manager not available: {e}")
    ARTIFACT_MANAGER_AVAILABLE = False


class TensorDetector:
    """CLI utility for detecting and organizing tensor recordings."""

    def __init__(self, storage_dir: Optional[str] = None, verbose: bool = False):
        """Initialize the tensor detector."""
        self.verbose = verbose

        if ARTIFACT_MANAGER_AVAILABLE:
            self.artifact_manager = ArtifactManager(base_storage_dir=storage_dir)
        else:
            self.artifact_manager = None

        # Common locations where tensor recordings might be found
        self.default_search_paths = [
            "/tmp",
            str(Path.cwd()),
            str(Path.home() / "Downloads"),
            "/var/tmp"
        ]

        # Enhanced patterns for tensor detection
        self.tensor_patterns = [
            "*.tensor",
            "*_tensor_*.bin",
            "*_recording_*.tensor",
            "tensor_data_*.bin",
            "*checkpoint*.tensor",
            "*l4ma*.tensor",
            "*embedding*.tensor",
            "*attention*.tensor",
            "*mlp*.tensor"
        ]

    def detect_recordings(
        self,
        search_paths: Optional[List[str]] = None,
        include_system_paths: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect tensor recordings in specified paths.

        Args:
            search_paths: Custom search paths
            include_system_paths: Whether to include default system paths

        Returns:
            List of detected recording information
        """
        if search_paths is None:
            search_paths = []

        if include_system_paths:
            search_paths.extend(self.default_search_paths)

        detected_recordings = []

        for search_path in search_paths:
            if self.verbose:
                print(f"🔍 Searching in: {search_path}")

            path_obj = Path(search_path)
            if not path_obj.exists():
                if self.verbose:
                    print(f"⚠️  Path does not exist: {search_path}")
                continue

            # Search using patterns
            for pattern in self.tensor_patterns:
                try:
                    for file_path in path_obj.rglob(pattern):
                        if file_path.is_file():
                            recording_info = self._analyze_tensor_file(file_path)
                            if recording_info:
                                detected_recordings.append(recording_info)
                                if self.verbose:
                                    print(f"✅ Found: {file_path.name} ({recording_info['size_mb']:.2f} MB)")
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Error searching with pattern {pattern}: {e}")

        return detected_recordings

    def _analyze_tensor_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a potential tensor file and extract metadata."""
        try:
            stat = file_path.stat()

            # Basic file analysis
            recording_info = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'detected_type': 'tensor_file',
                'confidence': 0.0
            }

            # Confidence scoring based on file characteristics
            confidence = 0.0

            # File extension analysis
            if file_path.suffix in ['.tensor', '.bin']:
                confidence += 0.3

            # Filename pattern analysis
            name_lower = file_path.name.lower()
            tensor_keywords = ['tensor', 'embedding', 'attention', 'mlp', 'checkpoint', 'l4ma']
            for keyword in tensor_keywords:
                if keyword in name_lower:
                    confidence += 0.2
                    break

            # Size analysis (reasonable tensor sizes)
            if 1024 < stat.st_size < 1024 * 1024 * 1024:  # 1KB to 1GB
                confidence += 0.2
            elif stat.st_size > 1024 * 1024 * 1024:  # > 1GB
                confidence += 0.1

            # File age analysis (recent files are more likely to be relevant)
            age_hours = (datetime.now() - recording_info['modified_time']).total_seconds() / 3600
            if age_hours < 24:  # Less than 24 hours old
                confidence += 0.2
            elif age_hours < 168:  # Less than a week old
                confidence += 0.1

            recording_info['confidence'] = min(confidence, 1.0)

            # Only return files with reasonable confidence
            if confidence >= 0.3:
                return recording_info

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error analyzing {file_path}: {e}")

        return None

    def organize_recordings(
        self,
        detected_recordings: List[Dict[str, Any]],
        session_name: str = "Auto-detected Session",
        model_name: str = "unknown",
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Organize detected recordings into the artifact management system.

        Args:
            detected_recordings: List of detected recording info
            session_name: Name for the session
            model_name: Model name for the session
            min_confidence: Minimum confidence threshold for organization

        Returns:
            Organization results
        """
        if not ARTIFACT_MANAGER_AVAILABLE:
            return {'error': 'Artifact manager not available'}

        # Filter by confidence
        high_confidence_recordings = [
            r for r in detected_recordings
            if r.get('confidence', 0) >= min_confidence
        ]

        if not high_confidence_recordings:
            return {
                'organized': 0,
                'total_detected': len(detected_recordings),
                'message': f'No recordings met minimum confidence threshold of {min_confidence}'
            }

        # Create session
        session_id = self.artifact_manager.create_session(
            session_name=f"{session_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            model_name=model_name,
            metadata={
                'auto_detected': True,
                'detection_time': datetime.now().isoformat(),
                'total_detected': len(detected_recordings),
                'organized_count': len(high_confidence_recordings)
            }
        )

        organized_count = 0
        organized_artifacts = []

        for recording_info in high_confidence_recordings:
            try:
                # Create a basic TensorRecording for organization
                # In practice, you might want more sophisticated metadata extraction
                tensor_recording = self._create_basic_tensor_recording(
                    recording_info, session_id
                )

                if tensor_recording:
                    artifact_id = self.artifact_manager.store_tensor_recording(
                        tensor_recording,
                        session_id,
                        tags=['auto_detected', f'confidence_{recording_info["confidence"]:.2f}']
                    )

                    organized_artifacts.append({
                        'artifact_id': artifact_id,
                        'original_path': recording_info['file_path'],
                        'confidence': recording_info['confidence']
                    })

                    organized_count += 1

            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to organize {recording_info['file_path']}: {e}")

        return {
            'session_id': session_id,
            'organized': organized_count,
            'total_detected': len(detected_recordings),
            'high_confidence': len(high_confidence_recordings),
            'artifacts': organized_artifacts,
            'success': organized_count > 0
        }

    def _create_basic_tensor_recording(
        self,
        recording_info: Dict[str, Any],
        session_id: int
    ) -> Optional[TensorRecording]:
        """Create a basic TensorRecording from detected file info."""
        try:
            import numpy as np

            file_path = Path(recording_info['file_path'])

            # Try to infer basic tensor properties
            # This is simplified - in practice you'd want more robust inference

            # Create basic metadata
            tensor_metadata = {
                'dtype': 'float32',  # Default assumption
                'shape': [1],  # Default shape
                'strides': [4],  # Default strides for float32
                'device': 'cpu',
                'memory_layout': 'C',
                'byte_order': '<'
            }

            # Basic device info
            device_info = {
                'platform': 'cpu',
                'device': 'cpu'
            }

            # Extract tensor name from filename
            tensor_name = file_path.stem

            # Create TensorRecording
            tensor_recording = TensorRecording(
                session_id=session_id,
                checkpoint_id=1,  # Default checkpoint
                tensor_name=tensor_name,
                tensor_metadata=tensor_metadata,
                tensor_data_path=str(file_path),
                backend_name='auto_detected',
                device_info=device_info,
                file_size_bytes=recording_info['size_bytes']
            )

            return tensor_recording

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not create TensorRecording for {recording_info['file_path']}: {e}")
            return None

    def generate_report(
        self,
        detected_recordings: List[Dict[str, Any]],
        organization_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a comprehensive detection and organization report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TENSOR RECORDING DETECTION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Detection summary
        report_lines.append("DETECTION SUMMARY:")
        report_lines.append(f"Total files detected: {len(detected_recordings)}")

        if detected_recordings:
            total_size = sum(r['size_mb'] for r in detected_recordings)
            avg_confidence = sum(r['confidence'] for r in detected_recordings) / len(detected_recordings)

            report_lines.append(f"Total size: {total_size:.2f} MB")
            report_lines.append(f"Average confidence: {avg_confidence:.2f}")

            # Confidence distribution
            high_conf = len([r for r in detected_recordings if r['confidence'] >= 0.7])
            med_conf = len([r for r in detected_recordings if 0.4 <= r['confidence'] < 0.7])
            low_conf = len([r for r in detected_recordings if r['confidence'] < 0.4])

            report_lines.append(f"High confidence (≥0.7): {high_conf}")
            report_lines.append(f"Medium confidence (0.4-0.7): {med_conf}")
            report_lines.append(f"Low confidence (<0.4): {low_conf}")

        report_lines.append("")

        # Organization summary
        if organization_result:
            report_lines.append("ORGANIZATION SUMMARY:")
            report_lines.append(f"Session ID: {organization_result.get('session_id', 'N/A')}")
            report_lines.append(f"Files organized: {organization_result.get('organized', 0)}")
            report_lines.append(f"Organization success: {organization_result.get('success', False)}")
            report_lines.append("")

        # Detailed file list
        if detected_recordings:
            report_lines.append("DETECTED FILES:")
            report_lines.append("-" * 40)

            # Sort by confidence descending
            sorted_recordings = sorted(
                detected_recordings,
                key=lambda x: x['confidence'],
                reverse=True
            )

            for i, recording in enumerate(sorted_recordings[:20]):  # Show top 20
                report_lines.append(
                    f"{i+1:2d}. {recording['file_name'][:40]:<40} "
                    f"{recording['size_mb']:6.2f}MB "
                    f"conf:{recording['confidence']:4.2f}"
                )

            if len(detected_recordings) > 20:
                report_lines.append(f"... and {len(detected_recordings) - 20} more files")

        return "\n".join(report_lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-detect and organize tensor recordings from L4MA inference"
    )

    parser.add_argument(
        'paths',
        nargs='*',
        help='Custom search paths (default: common system locations)'
    )

    parser.add_argument(
        '--storage-dir',
        help='Custom storage directory for artifact management'
    )

    parser.add_argument(
        '--session-name',
        default='Auto-detected Session',
        help='Name for the detection session'
    )

    parser.add_argument(
        '--model-name',
        default='unknown',
        help='Model name for the session'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for organization (0.0-1.0)'
    )

    parser.add_argument(
        '--no-organize',
        action='store_true',
        help='Only detect, do not organize into artifact system'
    )

    parser.add_argument(
        '--no-system-paths',
        action='store_true',
        help='Do not search default system paths'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--output-report',
        help='Save detection report to file'
    )

    args = parser.parse_args()

    # Initialize detector
    detector = TensorDetector(
        storage_dir=args.storage_dir,
        verbose=args.verbose
    )

    # Detect recordings
    print("🔍 Detecting tensor recordings...")
    detected_recordings = detector.detect_recordings(
        search_paths=args.paths if args.paths else None,
        include_system_paths=not args.no_system_paths
    )

    print(f"✅ Detected {len(detected_recordings)} potential tensor recordings")

    # Organize if requested
    organization_result = None
    if not args.no_organize and detected_recordings:
        print("📁 Organizing recordings...")
        organization_result = detector.organize_recordings(
            detected_recordings,
            session_name=args.session_name,
            model_name=args.model_name,
            min_confidence=args.min_confidence
        )

        if organization_result.get('success'):
            print(f"✅ Organized {organization_result['organized']} recordings into session {organization_result['session_id']}")
        else:
            print(f"⚠️  {organization_result.get('message', 'Organization failed')}")

    # Generate report
    report = detector.generate_report(detected_recordings, organization_result)

    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {args.output_report}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()