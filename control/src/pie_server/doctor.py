"""Doctor command for Pie Server CLI.

Implements: pie-server doctor
Runs environment health check to verify the system is ready for running.
"""

import importlib.metadata
import platform
import sys

import typer


def doctor() -> None:
    """Run environment health check.
    
    Checks Python version, GPU availability, and required dependencies
    to verify the system is ready for running the Pie backend.
    """
    typer.echo("Pie Backend Doctor")
    typer.echo("==================")
    
    # Check Python version
    python_version = sys.version.split()[0]
    typer.echo(f"Python version: {python_version}")
    if sys.version_info < (3, 11):
        typer.echo("  [FAIL] Python 3.11+ is required.")
    else:
        typer.echo("  [PASS] Python version is compatible.")

    # Check Platform
    typer.echo(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    # Check PyTorch
    try:
        import torch
        torch_version = torch.__version__
        typer.echo(f"PyTorch version: {torch_version}")
        typer.echo("  [PASS] PyTorch is installed.")
        
        # Check CUDA/MPS
        if torch.cuda.is_available():
            typer.echo(f"CUDA available: Yes (v{torch.version.cuda})")
            typer.echo(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                typer.echo(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        elif torch.backends.mps.is_available():
            typer.echo("MPS (Metal) available: Yes")
        else:
            typer.echo("CUDA/MPS available: No (Running on CPU)")
            
    except ImportError:
        typer.echo("  [FAIL] PyTorch is NOT installed.")
        typer.echo("\nInstall PyTorch: pip install torch")
        raise typer.Exit(1)

    # Check Dependencies
    typer.echo("\nDependencies:")
    
    # FlashInfer (CUDA)
    try:
        import flashinfer  # noqa: F401
        ver = importlib.metadata.version('flashinfer-python')
        typer.echo(f"  [PASS] flashinfer: Installed (v{ver})")
    except ImportError:
        typer.echo("  [WARN] flashinfer: Not installed (Required for CUDA performance)")
    except Exception as e:
        typer.echo(f"  [WARN] flashinfer: Error importing ({e})")

    # FBGEMM_GPU (CUDA)
    try:
        import fbgemm_gpu  # noqa: F401
        typer.echo(f"  [PASS] fbgemm_gpu: Installed (v{importlib.metadata.version('fbgemm-gpu-genai')})")
    except ImportError:
        try:
            ver = importlib.metadata.version('fbgemm-gpu-genai')
            typer.echo(f"  [PASS] fbgemm_gpu: Installed (v{ver})")
        except importlib.metadata.PackageNotFoundError:
            typer.echo("  [WARN] fbgemm_gpu: Not installed (Required for CUDA performance)")

    # PyObjC (Metal)
    if platform.system() == "Darwin":
        try:
            import objc  # noqa: F401
            typer.echo(f"  [PASS] pyobjc: Installed (v{importlib.metadata.version('pyobjc-core')})")
        except ImportError:
            typer.echo("  [WARN] pyobjc: Not installed (Required for Metal performance)")

    # Check pie-backend
    typer.echo("\nPie Backend:")
    try:
        ver = importlib.metadata.version('pie-backend')
        typer.echo(f"  [PASS] pie-backend: Installed (v{ver})")
    except importlib.metadata.PackageNotFoundError:
        typer.echo("  [WARN] pie-backend: Not installed")

    # Check pie-client
    try:
        ver = importlib.metadata.version('pie-client')
        typer.echo(f"  [PASS] pie-client: Installed (v{ver})")
    except importlib.metadata.PackageNotFoundError:
        typer.echo("  [WARN] pie-client: Not installed")

    typer.echo("\n✅ Doctor check complete.")
