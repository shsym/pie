"""
Hardware Detection Utilities

Detects the current hardware (Apple Silicon, NVIDIA GPU) for profiling purposes.
"""

import platform
import subprocess
import re
from typing import Optional, Dict, Any


def detect_apple_silicon() -> Optional[str]:
    """
    Detect Apple Silicon chip model.

    Returns:
        Chip identifier (e.g., "M4_Pro", "M3_Max") or None if not Apple Silicon
    """
    if platform.system() != "Darwin":
        return None

    try:
        # Use sysctl to get chip info
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        brand_string = result.stdout.strip()

        # Parse chip model from brand string
        # Examples: "Apple M4 Pro", "Apple M3 Max", "Apple M2"
        match = re.search(r"Apple (M\d+(?:\s+(?:Pro|Max|Ultra))?)", brand_string)
        if match:
            chip_name = match.group(1)
            # Convert to identifier format: "M4 Pro" -> "M4_Pro"
            return chip_name.replace(" ", "_")

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def detect_memory_size_gb() -> Optional[int]:
    """
    Detect total system memory in GB.

    Returns:
        Memory size in GB (rounded) or None if detection fails
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            memory_bytes = int(result.stdout.strip())
            # Convert to GB and round
            memory_gb = round(memory_bytes / (1024**3))
            return memory_gb
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass

    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Extract memory in kB
                        mem_kb = int(line.split()[1])
                        # Convert to GB and round
                        memory_gb = round(mem_kb / (1024**2))
                        return memory_gb
        except (IOError, ValueError, IndexError):
            pass

    return None


def detect_nvidia_gpu() -> Optional[str]:
    """
    Detect NVIDIA GPU model using nvidia-smi.

    Returns:
        GPU model identifier (e.g., "RTX_4090", "A100_80GB") or None if not NVIDIA
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_name = result.stdout.strip()

        # Map common GPU names to identifiers
        gpu_mapping = {
            r"RTX 4090": "RTX_4090",
            r"RTX 4080": "RTX_4080",
            r"RTX 3090": "RTX_3090",
            r"A100.*80GB": "A100_80GB",
            r"A100.*40GB": "A100_40GB",
            r"H100.*SXM": "H100_SXM",
            r"H100.*PCIe": "H100_PCIe",
            r"RTX A6000": "A6000",
            r"L4": "L4",
        }

        for pattern, identifier in gpu_mapping.items():
            if re.search(pattern, gpu_name, re.IGNORECASE):
                return identifier

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def detect_hardware() -> Dict[str, Any]:
    """
    Detect current hardware configuration.

    Returns:
        Dictionary with hardware information:
        - chip: Chip identifier or None
        - memory_gb: Total memory in GB or None
        - vendor: "apple" or "nvidia" or "unknown"
        - detected: Whether hardware was successfully detected
    """
    apple_chip = detect_apple_silicon()
    nvidia_gpu = detect_nvidia_gpu()
    memory_gb = detect_memory_size_gb()

    if apple_chip:
        return {
            "chip": apple_chip,
            "memory_gb": memory_gb,
            "vendor": "apple",
            "detected": True,
        }
    elif nvidia_gpu:
        return {
            "chip": nvidia_gpu,
            "memory_gb": memory_gb,
            "vendor": "nvidia",
            "detected": True,
        }
    else:
        return {
            "chip": None,
            "memory_gb": memory_gb,
            "vendor": "unknown",
            "detected": False,
        }


def get_hardware_info_for_profiling() -> Dict[str, Any]:
    """
    Get hardware information suitable for inclusion in profiling metadata.

    Returns:
        Dictionary with hardware info for profiling metadata
    """
    hw_info = detect_hardware()

    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "detected_chip": hw_info.get("chip"),
        "detected_memory_gb": hw_info.get("memory_gb"),
        "vendor": hw_info.get("vendor"),
    }
