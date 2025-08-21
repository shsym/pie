#!/usr/bin/env python
"""
Artifact Comparison Utility for PIE Metal Backend Validation

This script automatically compares Metal backend outputs with CUDA golden reference data,
providing detailed numerical analysis and validation reports.

Usage:
    python scripts/compare_artifacts.py --op OPERATION --case CASE_ID [options]
    python scripts/compare_artifacts.py --cuda-dir PATH --metal-dir PATH [options]

Examples:
    # Compare specific operation
    python scripts/compare_artifacts.py --op gemm --case test1

    # Compare all operations for a case
    python scripts/compare_artifacts.py --case production --all-ops

    # Custom directories
    python scripts/compare_artifacts.py --cuda-dir cuda-protocol-tests/tests/artifacts/gemm/test1 --metal-dir tests/artifacts/gemm/test1
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class ComparisonResult:
    """Results from comparing CUDA vs Metal artifacts"""
    operation: str
    case_id: str
    cuda_dir: Path
    metal_dir: Path
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    mean_rel_error: float
    matches_tolerance: bool
    tolerance_abs: float
    tolerance_rel: float
    file_comparisons: Dict[str, Dict[str, Any]]
    summary: str

class ArtifactComparator:
    """Compare CUDA and Metal artifacts with sophisticated numerical analysis"""

    # Default tolerances for cross-platform precision differences
    DEFAULT_ABS_TOLERANCE = 1e-3  # Absolute tolerance
    DEFAULT_REL_TOLERANCE = 1e-2  # Relative tolerance (1%)

    # Data type mappings
    DTYPE_MAP = {
        'fp32': np.float32,
        'fp16': np.float16,
        'bf16': None,  # bfloat16 requires special handling
        's32': np.int32,
        'u32': np.uint32,
        's64': np.int64,
        'u64': np.uint64
    }

    def __init__(self, abs_tolerance: Optional[float] = None, rel_tolerance: Optional[float] = None, verbose: bool = False):
        self.abs_tolerance = abs_tolerance or self.DEFAULT_ABS_TOLERANCE
        self.rel_tolerance = rel_tolerance or self.DEFAULT_REL_TOLERANCE
        self.verbose = verbose
        # Per-op, per-tensor, per-dtype tolerance overrides
        # Format: {(op, tensor_name, dtype): (abs_tol, rel_tol)}
        self.tolerance_overrides: Dict[Tuple[str, str, str], Tuple[float, float]] = {
            # gemm fp32 output can differ slightly due to accumulation order; allow small abs diff
            ('gemm', 'C', 'fp32'): (3.0e-2, 1.0e-2),
            # rms_norm bf16 output is sensitive due to bf16 resolution and reduction/rsqrt variance
            ('rms_norm', 'output', 'bf16'): (self.DEFAULT_ABS_TOLERANCE, 1.5e-2),
            # rope outputs involve sin/cos rotations in bf16 which vary slightly across backends
            # One bf16 ULP around 1.0 is 7.8125e-3; allow up to 1e-2 abs for RoPE outputs in bf16
            ('rope', 'q_output', 'bf16'): (1e-2, 2e-2),
            ('rope', 'k_output', 'bf16'): (1e-2, 2e-2),
            # batch_prefill_attention uses fp16 kernels with bf16 I/O bridging; expect up to one bf16 ULP
            # in outputs after fp16 round-trip. Allow abs up to 1e-2 for bf16 outputs.
            ('batch_prefill_attention', 'output', 'bf16'): (1e-2, 2e-2),
            # grouped_gemm accumulates in bf16; permit up to 1 ULP (~3.125e-2) and modest rel error
            ('grouped_gemm', 'C', 'bf16'): (3.2e-2, 1.5e-2),
        }

    def load_meta_json(self, artifact_dir: Path) -> Dict[str, Any]:
        """Load metadata from meta.json file"""
        meta_path = artifact_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {artifact_dir}")

        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_binary_tensor(self, file_path: Path, dtype: str, shape: List[int]) -> np.ndarray:
        """Load binary tensor data with proper dtype conversion"""
        if not file_path.exists():
            raise FileNotFoundError(f"Binary file not found: {file_path}")

        # Handle bfloat16 (stored as uint16, needs conversion)
        if dtype == 'bf16':
            raw_data = np.fromfile(file_path, dtype=np.uint16)
            # Convert bfloat16 to float32 for comparison
            # bfloat16 is stored in upper 16 bits of float32
            expanded = raw_data.astype(np.uint32) << 16
            data = expanded.view(np.float32)
        else:
            np_dtype = self.DTYPE_MAP.get(dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported dtype: {dtype}")
            data = np.fromfile(file_path, dtype=np_dtype)

        # Reshape to expected shape
        try:
            return data.reshape(shape)
        except ValueError as e:
            actual_size = data.size
            expected_size = np.prod(shape)
            raise ValueError(
                f"Shape mismatch for {file_path}: expected {expected_size} elements "
                f"(shape {shape}), got {actual_size} elements"
            ) from e

    def compare_tensors(self, cuda_tensor: np.ndarray, metal_tensor: np.ndarray,
                       tensor_name: str) -> Dict[str, Any]:
        """Compare two tensors with comprehensive numerical analysis"""
        if cuda_tensor.shape != metal_tensor.shape:
            return {
                'status': 'SHAPE_MISMATCH',
                'cuda_shape': cuda_tensor.shape,
                'metal_shape': metal_tensor.shape,
                'error': f"Shape mismatch: CUDA {cuda_tensor.shape} vs Metal {metal_tensor.shape}"
            }

        # Convert to float64 for high-precision comparison
        cuda_f64 = cuda_tensor.astype(np.float64)
        metal_f64 = metal_tensor.astype(np.float64)

        # Handle infinities: when both values are the same infinity, treat as exact match (zero error)
        same_inf_mask = np.isinf(cuda_f64) & np.isinf(metal_f64) & (np.sign(cuda_f64) == np.sign(metal_f64))

        # Compute absolute difference; set positions of same-signed infinities to zero explicitly
        raw_abs_diff = np.abs(cuda_f64 - metal_f64)
        abs_diff = np.where(same_inf_mask, 0.0, raw_abs_diff)

        # Avoid division by zero in relative error; treat same-signed infinities as zero relative error
        cuda_abs = np.abs(cuda_f64)
        raw_rel = np.where(cuda_abs > 1e-10, abs_diff / np.where(cuda_abs > 1e-10, cuda_abs, 1.0), 0.0)
        rel_diff = np.where(same_inf_mask, 0.0, raw_rel)

        # Statistics
        max_abs_error = np.max(abs_diff)
        max_rel_error = np.max(rel_diff)
        mean_abs_error = np.mean(abs_diff)
        mean_rel_error = np.mean(rel_diff)

        # Check tolerance
        abs_ok = max_abs_error <= self.abs_tolerance
        rel_ok = max_rel_error <= self.rel_tolerance
        passes_tolerance = abs_ok or rel_ok

        # Find worst error locations for debugging
        worst_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        worst_rel_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)

        result = {
            'status': 'PASS' if passes_tolerance else 'FAIL',
            'max_abs_error': float(max_abs_error),
            'max_rel_error': float(max_rel_error),
            'mean_abs_error': float(mean_abs_error),
            'mean_rel_error': float(mean_rel_error),
            'abs_tolerance_pass': abs_ok,
            'rel_tolerance_pass': rel_ok,
            'shape': cuda_tensor.shape,
            'total_elements': cuda_tensor.size,
            'worst_abs_location': worst_abs_idx,
            'worst_rel_location': worst_rel_idx,
            'worst_abs_values': {
                'cuda': float(cuda_f64[worst_abs_idx]),
                'metal': float(metal_f64[worst_abs_idx]),
                'diff': float(abs_diff[worst_abs_idx])
            },
            'worst_rel_values': {
                'cuda': float(cuda_f64[worst_rel_idx]),
                'metal': float(metal_f64[worst_rel_idx]),
                'diff': float(rel_diff[worst_rel_idx])
            }
        }

        if self.verbose:
            print(f"  {tensor_name}:")
            print(f"    Shape: {cuda_tensor.shape}")
            print(f"    Max absolute error: {max_abs_error:.2e} (tolerance: {self.abs_tolerance:.2e}) {'✅' if abs_ok else '❌'}")
            print(f"    Max relative error: {max_rel_error:.2e} (tolerance: {self.rel_tolerance:.2e}) {'✅' if rel_ok else '❌'}")
            print(f"    Status: {'✅ PASS' if passes_tolerance else '❌ FAIL'}")

        return result

    def compare_artifacts(self, cuda_dir: Path, metal_dir: Path) -> ComparisonResult:
        """Compare all artifacts between CUDA and Metal directories"""
        # Load metadata
        try:
            cuda_meta = self.load_meta_json(cuda_dir)
            _ = self.load_meta_json(metal_dir)
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load metadata: {e}") from e

        operation = cuda_meta.get('op', 'unknown')
        case_id = cuda_meta.get('case_id', 'unknown')

        if self.verbose:
            print(f"\n=== Comparing {operation} (case: {case_id}) ===")
            print(f"CUDA dir: {cuda_dir}")
            print(f"Metal dir: {metal_dir}")

        # Get tensor info from metadata
        dtype_map = cuda_meta.get('dtype_map', {})
        shape_map = cuda_meta.get('shape_map', {})

        file_comparisons = {}
        all_max_abs_errors = []
        all_max_rel_errors = []
        all_mean_abs_errors = []
        all_mean_rel_errors = []
        overall_pass = True

        # Compare each binary tensor file
        for tensor_name, dtype in dtype_map.items():
            if tensor_name not in shape_map:
                continue

            shape = shape_map[tensor_name]
            cuda_file = cuda_dir / f"{tensor_name}.bin"
            metal_file = metal_dir / f"{tensor_name}.bin"

            try:
                cuda_tensor = self.load_binary_tensor(cuda_file, dtype, shape)
                metal_tensor = self.load_binary_tensor(metal_file, dtype, shape)

                # Apply scoped tolerance overrides when available
                orig_abs_tol, orig_rel_tol = self.abs_tolerance, self.rel_tolerance
                override_key = (operation, tensor_name, dtype)
                if override_key in self.tolerance_overrides:
                    self.abs_tolerance, self.rel_tolerance = self.tolerance_overrides[override_key]
                    if self.verbose:
                        print(
                            f"  Using override tolerances for {tensor_name} (op={operation}, dtype={dtype}): "
                            f"abs={self.abs_tolerance:.2e}, rel={self.rel_tolerance:.2e}"
                        )

                comparison = self.compare_tensors(cuda_tensor, metal_tensor, tensor_name)

                # Restore original tolerances for next tensors
                self.abs_tolerance, self.rel_tolerance = orig_abs_tol, orig_rel_tol
                file_comparisons[tensor_name] = comparison

                if comparison['status'] == 'PASS':
                    all_max_abs_errors.append(comparison['max_abs_error'])
                    all_max_rel_errors.append(comparison['max_rel_error'])
                    all_mean_abs_errors.append(comparison['mean_abs_error'])
                    all_mean_rel_errors.append(comparison['mean_rel_error'])
                else:
                    overall_pass = False

            except (FileNotFoundError, ValueError) as e:
                file_comparisons[tensor_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                overall_pass = False
                if self.verbose:
                    print(f"  {tensor_name}: ❌ ERROR - {e}")

        # Compute overall statistics
        if all_max_abs_errors:
            overall_max_abs = float(max(all_max_abs_errors))
            overall_max_rel = float(max(all_max_rel_errors))
            overall_mean_abs = float(np.mean(all_mean_abs_errors))
            overall_mean_rel = float(np.mean(all_mean_rel_errors))
        else:
            overall_max_abs = float('inf')
            overall_max_rel = float('inf')
            overall_mean_abs = float('inf')
            overall_mean_rel = float('inf')
            overall_pass = False

        # Generate summary
        if overall_pass:
            summary = "✅ PASS - All tensors match within tolerance"
        else:
            failed_tensors = [name for name, comp in file_comparisons.items()
                            if comp['status'] != 'PASS']
            summary = f"❌ FAIL - {len(failed_tensors)} tensor(s) failed: {', '.join(failed_tensors)}"

        return ComparisonResult(
            operation=operation,
            case_id=case_id,
            cuda_dir=cuda_dir,
            metal_dir=metal_dir,
            max_abs_error=overall_max_abs,
            max_rel_error=overall_max_rel,
            mean_abs_error=overall_mean_abs,
            mean_rel_error=overall_mean_rel,
            matches_tolerance=overall_pass,
            tolerance_abs=self.abs_tolerance,
            tolerance_rel=self.rel_tolerance,
            file_comparisons=file_comparisons,
            summary=summary
        )

def find_artifact_directories(base_dir: Path, operation: str, case_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find CUDA and Metal artifact directories for an operation and case"""
    cuda_dir = None
    metal_dir = None

    # Look for CUDA artifacts (no suffix or _cuda suffix)
    op_dir = base_dir / operation
    if op_dir.exists():
        for candidate in [case_id, f"{case_id}_cuda"]:
            candidate_dir = op_dir / candidate
            if candidate_dir.exists() and (candidate_dir / "meta.json").exists():
                cuda_dir = candidate_dir
                break

    # Look for Metal artifacts (_metal suffix)
    metal_candidate = op_dir / f"{case_id}_metal"
    if metal_candidate.exists() and (metal_candidate / "meta.json").exists():
        metal_dir = metal_candidate

    return cuda_dir, metal_dir

def print_detailed_report(result: ComparisonResult):
    """Print detailed comparison report"""
    print(f"\n{'='*60}")
    print(f"COMPARISON REPORT: {result.operation} (case: {result.case_id})")
    print(f"{'='*60}")
    print(f"CUDA artifacts:  {result.cuda_dir}")
    print(f"Metal artifacts: {result.metal_dir}")
    print(f"\nOverall Result: {result.summary}")
    print("\nTolerance Settings:")
    print(f"  Absolute: {result.tolerance_abs:.2e}")
    print(f"  Relative: {result.tolerance_rel:.2e}")
    print("\nOverall Statistics:")
    print(f"  Max absolute error: {result.max_abs_error:.2e}")
    print(f"  Max relative error: {result.max_rel_error:.2e}")
    print(f"  Mean absolute error: {result.mean_abs_error:.2e}")
    print(f"  Mean relative error: {result.mean_rel_error:.2e}")

    print("\nPer-Tensor Results:")
    for tensor_name, comparison in result.file_comparisons.items():
        status_icon = "✅" if comparison['status'] == 'PASS' else "❌"
        print(f"  {status_icon} {tensor_name}: {comparison['status']}")

        if comparison['status'] == 'PASS':
            # Ensure Python bools (values may be numpy.bool_)
            abs_ok = bool(comparison.get('abs_tolerance_pass', False))
            rel_ok = bool(comparison.get('rel_tolerance_pass', False))
            # Always show the absolute and relative errors
            print(f"    Max abs error: {comparison['max_abs_error']:.2e}")
            # If relative error is out of range but absolute error is within tolerance,
            # annotate to avoid confusion when overall status is PASS due to abs tolerance
            if (not rel_ok) and abs_ok:
                print(f"    Max rel error: {comparison['max_rel_error']:.2e} (out of range but abs is within tol)")
            else:
                print(f"    Max rel error: {comparison['max_rel_error']:.2e}")
        elif comparison['status'] == 'ERROR':
            print(f"    Error: {comparison['error']}")
        elif comparison['status'] == 'SHAPE_MISMATCH':
            print(f"    CUDA shape: {comparison['cuda_shape']}")
            print(f"    Metal shape: {comparison['metal_shape']}")
        elif comparison['status'] == 'FAIL':
            # Provide detailed error ranges for failed tensors
            max_abs = comparison.get('max_abs_error', float('nan'))
            max_rel = comparison.get('max_rel_error', float('nan'))
            mean_abs = comparison.get('mean_abs_error', float('nan'))
            mean_rel = comparison.get('mean_rel_error', float('nan'))
            abs_tol_ok = comparison.get('abs_tolerance_pass', False)
            rel_tol_ok = comparison.get('rel_tolerance_pass', False)

            print(f"    Max abs error: {max_abs:.2e} (<= tol? {'yes' if abs_tol_ok else 'no'})")
            print(f"    Max rel error: {max_rel:.2e} (<= tol? {'yes' if rel_tol_ok else 'no'})")
            print(f"    Mean abs error: {mean_abs:.2e}")
            print(f"    Mean rel error: {mean_rel:.2e}")

            worst_abs_idx = comparison.get('worst_abs_location')
            worst_rel_idx = comparison.get('worst_rel_location')
            worst_abs_vals = comparison.get('worst_abs_values', {})
            worst_rel_vals = comparison.get('worst_rel_values', {})
            if worst_abs_idx is not None and worst_abs_vals:
                print(f"    Worst abs @ {tuple(worst_abs_idx)}: cuda={worst_abs_vals.get('cuda'):.4e}, "
                      f"metal={worst_abs_vals.get('metal'):.4e}, diff={worst_abs_vals.get('diff'):.4e}")
            if worst_rel_idx is not None and worst_rel_vals:
                print(f"    Worst rel @ {tuple(worst_rel_idx)}: cuda={worst_rel_vals.get('cuda'):.4e}, "
                      f"metal={worst_rel_vals.get('metal'):.4e}, diff={worst_rel_vals.get('diff'):.4e}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Metal backend outputs with CUDA golden reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input specification (either op+case or direct directories)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--op', help='Operation name (e.g., gemm, softmax)')
    input_group.add_argument('--cuda-dir', type=Path, help='Direct path to CUDA artifacts directory')

    parser.add_argument('--case', help='Case ID (required with --op)', default='test1')
    parser.add_argument('--metal-dir', type=Path, help='Direct path to Metal artifacts directory (for --cuda-dir mode)')
    parser.add_argument('--all-ops', action='store_true', help='Compare all operations for the given case')

    # Base directories
    parser.add_argument('--cuda-base', type=Path, default=Path('cuda-protocol-tests/tests/artifacts'),
                       help='Base directory for CUDA artifacts')
    parser.add_argument('--metal-base', type=Path, default=Path('tests/artifacts'),
                       help='Base directory for Metal artifacts')

    # Tolerance settings
    parser.add_argument('--abs-tolerance', type=float, default=1e-3,
                       help='Absolute error tolerance (default: 1e-3)')
    parser.add_argument('--rel-tolerance', type=float, default=1e-2,
                       help='Relative error tolerance (default: 1e-2)')

    # Output options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json-output', type=Path, help='Save results to JSON file')
    parser.add_argument('--summary-only', action='store_true', help='Show summary only')

    args = parser.parse_args()

    comparator = ArtifactComparator(
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
        verbose=args.verbose
    )

    results = []

    # Determine comparison mode
    if args.cuda_dir:
        # Direct path mode
        cuda_dir = args.cuda_dir
        if not cuda_dir.exists():
            print(f"Error: CUDA artifacts directory does not exist: {cuda_dir}")
            sys.exit(1)

        if args.metal_dir:
            metal_dir = args.metal_dir
        else:
            # Heuristic: infer metal directory from CUDA directory name
            base = cuda_dir.parent
            case_name = cuda_dir.name
            if case_name.endswith('_cuda'):
                metal_case = case_name[:-5] + '_metal'
            else:
                metal_case = case_name + '_metal'
            metal_dir = base / metal_case

        if not metal_dir.exists():
            print(f"Error: Metal artifacts not found: {metal_dir}")
            sys.exit(1)

        result = comparator.compare_artifacts(cuda_dir, metal_dir)
        results.append(result)
    else:
        # Single operation comparison using base directories
        cuda_dir, metal_dir = find_artifact_directories(args.metal_base, args.op, args.case)

        # Also check cuda-base if not found in metal-base
        if not cuda_dir and args.cuda_base.exists():
            cuda_dir, _ = find_artifact_directories(args.cuda_base, args.op, args.case)

        if not cuda_dir:
            print(f"Error: CUDA artifacts not found for {args.op}/{args.case}")
            print(f"Searched in: {args.cuda_base} and {args.metal_base}")
            sys.exit(1)

        if not metal_dir:
            print(f"Error: Metal artifacts not found for {args.op}/{args.case}")
            print(f"Expected: {args.metal_base / args.op / f'{args.case}_metal'}")
            sys.exit(1)

        result = comparator.compare_artifacts(cuda_dir, metal_dir)
        results.append(result)

    # Output results
    if not results:
        print("No comparisons performed")
        sys.exit(1)

    overall_pass = all(r.matches_tolerance for r in results)

    if args.summary_only:
        print(f"\nSUMMARY: {len(results)} comparison(s)")
        for result in results:
            status = "✅ PASS" if result.matches_tolerance else "❌ FAIL"
            print(f"  {status} {result.operation}/{result.case_id}")
        print(f"\nOverall: {'✅ ALL PASS' if overall_pass else '❌ SOME FAILED'}")
    else:
        for result in results:
            print_detailed_report(result)

    # Save JSON output if requested
    if args.json_output:
        json_data = {
            'summary': {
                'total_comparisons': len(results),
                'passed': sum(1 for r in results if r.matches_tolerance),
                'failed': sum(1 for r in results if not r.matches_tolerance),
                'overall_pass': overall_pass
            },
            'results': []
        }

        for result in results:
            json_data['results'].append({
                'operation': result.operation,
                'case_id': result.case_id,
                'cuda_dir': str(result.cuda_dir),
                'metal_dir': str(result.metal_dir),
                'max_abs_error': result.max_abs_error,
                'max_rel_error': result.max_rel_error,
                'mean_abs_error': result.mean_abs_error,
                'mean_rel_error': result.mean_rel_error,
                'matches_tolerance': result.matches_tolerance,
                'tolerance_abs': result.tolerance_abs,
                'tolerance_rel': result.tolerance_rel,
                'file_comparisons': result.file_comparisons,
                'summary': result.summary
            })

        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to: {args.json_output}")

    # Exit with appropriate code
    sys.exit(0 if overall_pass else 1)

if __name__ == '__main__':
    main()