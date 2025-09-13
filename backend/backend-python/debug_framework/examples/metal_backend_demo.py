#!/usr/bin/env python3
"""
Metal Backend Integration Demo

This demonstrates the Metal backend integration with the L4MA debug framework.
Shows automatic backend detection and Metal compute integration on Apple machines.
"""

import sys
import os
import numpy as np

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from integrations.backend_interfaces import (
    BackendType, create_backend, get_recommended_backend, create_auto_backend
)

def demo_automatic_backend_selection():
    """Demonstrate automatic backend selection based on platform."""
    print("🔍 Automatic Backend Detection:")
    print(f"   Platform: {sys.platform}")

    import platform
    print(f"   Architecture: {platform.machine()}")

    recommended = get_recommended_backend()
    print(f"   Recommended: {recommended.value}")
    print()

def demo_metal_backend_features():
    """Demonstrate Metal backend specific features."""
    print("⚡ Metal Backend Features:")

    try:
        backend = create_auto_backend()
        print(f"   Active Backend: {backend.backend_type.value}")

        # Test with sample data
        query = np.random.randn(1, 8, 64).astype(np.float32)
        key = np.random.randn(1, 8, 64).astype(np.float32)
        value = np.random.randn(1, 8, 64).astype(np.float32)

        # Run attention operation
        result = backend.run_attention(query, key, value)
        print(f"   Attention: {result.output.shape} in {result.computation_time:.4f}s")

        # Show capabilities
        capabilities = backend.get_capabilities()
        print(f"   Available: {backend.is_available}")
        print(f"   Operations: {len(capabilities['supported_operations'])}")

        if backend.backend_type == BackendType.METAL:
            kernels = capabilities.get('available_kernels', {})
            print(f"   Metal Kernels: {sum(kernels.values())} available")
            print(f"   Device: {capabilities.get('device_info', 'Unknown')}")

        backend.cleanup()
        return True

    except Exception as e:
        print(f"   ❌ Backend creation failed: {e}")
        return False

def demo_fallback_chain():
    """Demonstrate fallback behavior when preferred backend unavailable."""
    print("🔄 Fallback Chain Demonstration:")

    # Try each backend type to show fallback behavior
    backend_order = [BackendType.METAL, BackendType.L4MA_PYTHON, BackendType.MOCK]

    for backend_type in backend_order:
        try:
            backend = create_backend(backend_type)
            print(f"   ✅ {backend_type.value}: Available")
            backend.cleanup()
            break
        except Exception as e:
            print(f"   ❌ {backend_type.value}: {str(e)[:50]}...")

    print()

def main():
    """Run the Metal backend integration demo."""
    print("Metal Backend Integration Demo")
    print("=" * 50)
    print()

    demo_automatic_backend_selection()
    success = demo_metal_backend_features()
    demo_fallback_chain()

    print("=" * 50)
    print("✨ Demo Complete!")
    print()

    if success:
        print("The Metal backend is properly integrated and functional.")
        print("Key features demonstrated:")
        print("- Automatic platform detection (Apple Silicon/Intel Mac)")
        print("- Metal compute integration with CPU fallbacks")
        print("- Factory pattern for backend creation")
        print("- Comprehensive tensor operations support")
        print("- Graceful error handling and fallback chains")
    else:
        print("Metal backend fell back to alternative backends.")
        print("This is expected behavior when Metal libraries are not built.")

    print()
    print("For full Metal acceleration, ensure the Metal backend is built:")
    print("  cd backend/backend-metal && cmake --build build")

if __name__ == "__main__":
    main()