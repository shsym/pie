"""Inferlet subcommand implementation for the Pie CLI.

This module implements the `pie-client inferlet` subcommand group for
searching, querying, and publishing inferlets to the Pie Registry.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Optional

import toml
import typer

from .config import ConfigFile
from . import path as path_utils
from .registry import (
    REGISTRY_URL,
    RegistryClient,
    RegistryError,
    PublishStartRequest,
    PublishCommitRequest,
    resolve_name,
)


# Create the inferlet subcommand group
inferlet_app = typer.Typer(
    name="inferlet",
    help="Manage inferlet packages from the Pie Registry.",
    no_args_is_help=True,
)


def _get_token() -> Optional[str]:
    """Load the registry token from config if available."""
    config_path = path_utils.get_default_config_path()
    if config_path.exists():
        config = ConfigFile.load(config_path)
        return config.registry_token
    return None


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_downloads(downloads: int) -> str:
    """Format download count as human-readable string."""
    if downloads < 1000:
        return str(downloads)
    elif downloads < 1_000_000:
        return f"{downloads / 1000:.1f}k"
    else:
        return f"{downloads / 1_000_000:.1f}M"


@inferlet_app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Search query.")] = "",
    page: Annotated[int, typer.Option("--page", "-p", help="Page number.")] = 1,
    per_page: Annotated[int, typer.Option("--per-page", help="Results per page.")] = 20,
    namespace: Annotated[Optional[str], typer.Option("--namespace", "-n", help="Filter by namespace.")] = None,
) -> None:
    """Search for inferlets in the registry."""
    try:
        with RegistryClient(base_url=REGISTRY_URL) as client:
            result = client.search(query=query, page=page, per_page=per_page, namespace=namespace)
            
            if result.total == 0:
                if query:
                    typer.echo(f"No inferlets found matching '{query}'")
                else:
                    typer.echo("No inferlets found")
                return
            
            typer.echo(f"Found {result.total} inferlet(s) (page {result.page}/{result.pages}):")
            typer.echo()
            
            # Print header
            typer.echo(f"{'NAME':<30} {'VERSION':<10} {'DOWNLOADS':<10} DESCRIPTION")
            typer.echo("-" * 80)
            
            for item in result.items:
                name = item.full_name[:28] + ".." if len(item.full_name) > 30 else item.full_name
                version = item.latest_version or "-"
                downloads = _format_downloads(item.downloads)
                description = (item.description or "")[:30]
                if item.description and len(item.description) > 30:
                    description += "..."
                
                typer.echo(f"{name:<30} {version:<10} {downloads:<10} {description}")
            
            if result.pages > 1:
                typer.echo()
                typer.echo(f"Use --page to see more results (page {result.page}/{result.pages})")
                
    except RegistryError as e:
        typer.echo(f"❌ Error: {e.detail}", err=True)
        raise typer.Exit(1)


@inferlet_app.command("info")
def info(
    name: Annotated[str, typer.Argument(help="Inferlet name (e.g., 'react' or 'ingim/tree-of-thought').")],
) -> None:
    """Get detailed information about an inferlet."""
    try:
        with RegistryClient(base_url=REGISTRY_URL) as client:
            detail = client.info(name)
            
            typer.echo(f"📦 {detail.full_name}")
            typer.echo()
            typer.echo(f"   Downloads: {_format_downloads(detail.downloads)}")
            typer.echo(f"   Created: {detail.created_at.strftime('%Y-%m-%d')}")
            typer.echo()
            
            if detail.versions:
                typer.echo("   Versions:")
                for v in detail.versions[:10]:  # Show latest 10
                    yanked = " (yanked)" if v.yanked else ""
                    size = _format_size(v.size_bytes)
                    typer.echo(f"      {v.num:<12} {size:<10}{yanked}")
                    if v.description:
                        typer.echo(f"                     {v.description[:50]}")
                
                if len(detail.versions) > 10:
                    typer.echo(f"      ... and {len(detail.versions) - 10} more")
                
                # Show interface spec of latest version
                latest = detail.versions[0]
                if latest.interface_spec:
                    typer.echo()
                    typer.echo("   Interface:")
                    if "inputs" in latest.interface_spec:
                        typer.echo("      Inputs:")
                        for inp in latest.interface_spec["inputs"]:
                            inp_name = inp.get("name", "?")
                            inp_type = inp.get("type", "?")
                            typer.echo(f"         - {inp_name}: {inp_type}")
                    if "outputs" in latest.interface_spec:
                        typer.echo("      Outputs:")
                        for out in latest.interface_spec["outputs"]:
                            out_name = out.get("name", "?")
                            out_type = out.get("type", "?")
                            typer.echo(f"         - {out_name}: {out_type}")
                
                if latest.requires_engine:
                    typer.echo()
                    typer.echo(f"   Requires engine: {latest.requires_engine}")
            else:
                typer.echo("   No versions published yet")
                
    except RegistryError as e:
        if e.status_code == 404:
            namespace, pkg_name = resolve_name(name)
            typer.echo(f"❌ Inferlet '{namespace}/{pkg_name}' not found", err=True)
        else:
            typer.echo(f"❌ Error: {e.detail}", err=True)
        raise typer.Exit(1)


@inferlet_app.command("publish")
def publish(
    directory: Annotated[Path, typer.Argument(help="Directory containing Pie.toml manifest.")] = Path("."),
) -> None:
    """Publish the inferlet in the specified directory.
    
    Requires a pie.toml manifest file and a .wasm artifact.
    You must be logged in with `pie-client login` first.
    """
    directory = directory.expanduser().resolve()
    
    # Check for Pie.toml
    manifest_path = directory / "Pie.toml"
    if not manifest_path.exists():
        typer.echo(f"❌ No Pie.toml found in {directory}", err=True)
        raise typer.Exit(1)
    
    # Load the manifest
    try:
        manifest = toml.load(manifest_path)
    except toml.TomlDecodeError as e:
        typer.echo(f"❌ Failed to parse Pie.toml: {e}", err=True)
        raise typer.Exit(1)
    
    # Extract package info
    package = manifest.get("package", {})
    full_name = package.get("name")
    version = package.get("version")
    description = package.get("description")
    
    if not full_name:
        typer.echo("❌ Missing 'package.name' in Pie.toml", err=True)
        raise typer.Exit(1)
    
    if not version:
        typer.echo("❌ Missing 'package.version' in Pie.toml", err=True)
        raise typer.Exit(1)
    
    # Parse namespace/name
    namespace, name = resolve_name(full_name)
    
    # Get engine requirements
    engine = manifest.get("engine", {})
    requires_engine = engine.get("min_version")
    
    # Get interface spec
    interface = manifest.get("interface", {})
    interface_spec = interface if interface else None
    
    # Find the .wasm artifact
    wasm_path = directory / f"{name}.wasm"
    if not wasm_path.exists():
        # Try common alternatives
        for pattern in ["*.wasm", "target/wasm32-wasip2/release/*.wasm", "target/*.wasm"]:
            matches = list(directory.glob(pattern))
            if matches:
                wasm_path = matches[0]
                break
    
    if not wasm_path.exists():
        typer.echo(f"❌ No .wasm artifact found. Expected: {name}.wasm", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"📦 Publishing {namespace}/{name}@{version}")
    typer.echo(f"   Artifact: {wasm_path.name}")
    
    # Load the token
    token = _get_token()
    if not token:
        typer.echo()
        typer.echo("❌ Not authenticated. Run `pie-client login` first.", err=True)
        raise typer.Exit(1)
    
    # Read and hash the artifact
    artifact_bytes = wasm_path.read_bytes()
    checksum = hashlib.sha256(artifact_bytes).hexdigest()
    size_bytes = len(artifact_bytes)
    
    typer.echo(f"   Size: {_format_size(size_bytes)}")
    typer.echo(f"   Checksum: {checksum[:16]}...")
    typer.echo()
    
    try:
        with RegistryClient(token=token, base_url=REGISTRY_URL) as client:
            # Verify we're authenticated as the right user
            user = client.get_me()
            
            if namespace != "std" and namespace != user.login:
                typer.echo(f"❌ Cannot publish to namespace '{namespace}' as user '{user.login}'", err=True)
                raise typer.Exit(1)
            
            if namespace == "std" and not user.is_superuser:
                typer.echo("❌ Only superusers can publish to the 'std' namespace", err=True)
                raise typer.Exit(1)
            
            typer.echo(f"🔐 Publishing as: {user.login}")
            typer.echo()
            
            # Start the publish process
            typer.echo("📤 Starting publish...")
            start_req = PublishStartRequest(
                namespace=namespace,
                name=name,
                version=version,
                checksum=checksum,
                size_bytes=size_bytes,
                description=description,
                requires_engine=requires_engine,
                interface_spec=interface_spec,
            )
            start_resp = client.start_publish(start_req)
            
            typer.echo("📤 Uploading artifact...")
            client.upload_artifact(start_resp.upload_url, artifact_bytes)
            
            typer.echo("📤 Finalizing publish...")
            commit_req = PublishCommitRequest(
                namespace=namespace,
                name=name,
                version=version,
                storage_path=start_resp.storage_path,
            )
            commit_resp = client.commit_publish(commit_req)
            
            typer.echo()
            typer.echo(f"✅ Published: {commit_resp.full_name}@{commit_resp.version}")
            typer.echo()
            typer.echo("   Install with:")
            typer.echo(f"      pie run {commit_resp.full_name}")
            
    except RegistryError as e:
        typer.echo(f"❌ Publish failed: {e.detail}", err=True)
        raise typer.Exit(1)


def handle_inferlet_search(
    query: str = "",
    page: int = 1,
    per_page: int = 20,
    namespace: Optional[str] = None,
) -> None:
    """Wrapper for programmatic access to search."""
    search(query=query, page=page, per_page=per_page, namespace=namespace)


def handle_inferlet_info(name: str) -> None:
    """Wrapper for programmatic access to info."""
    info(name=name)


def handle_inferlet_publish(directory: Path = Path(".")) -> None:
    """Wrapper for programmatic access to publish."""
    publish(directory=directory)
