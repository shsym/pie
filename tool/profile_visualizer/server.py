"""HTTP server for profiling visualizer."""

import json
import os
import urllib.parse
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path


class ProfileVisualizerHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve profiling visualization."""

    profiling_data = None
    search_directory = None
    root_directory = None
    static_dir = None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif self.path == "/styles.css":
            self._serve_file("styles.css", "text/css")
        elif self.path == "/app.js":
            self._serve_file("app.js", "application/javascript")
        elif self.path == "/data.json":
            self._serve_json(self.profiling_data)
        elif self.path.startswith("/list_files"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            base_path = query.get("path", [""])[0]
            if not base_path:
                base_path = self.search_directory
            files = self._list_json_files(base_path)
            self._serve_json(files)
        elif self.path.startswith("/list_dirs"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            base_path = query.get("path", [""])[0]
            if not base_path:
                base_path = self.search_directory
            dirs = self._list_directories(base_path)
            self._serve_json(dirs)
        elif self.path.startswith("/load_file?"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            filepath = query.get("path", [""])[0]
            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._serve_json(data)
                except Exception as e:  # pylint: disable=broad-except
                    self.send_error(400, f"Error loading file: {str(e)}")
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "File not found")

    def _serve_file(self, filename, content_type):
        """Serve a static file from the static directory."""
        filepath = os.path.join(self.static_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", content_type)
            self.end_headers()
            self.wfile.write(content.encode())
        except FileNotFoundError:
            self.send_error(404, f"File not found: {filename}")

    def _serve_json(self, data):
        """Serve JSON data."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _list_json_files(self, base_path=""):
        """List JSON files in the specified directory (non-recursive)."""
        files = []
        if not base_path:
            base_path = self.search_directory

        print(f"[DEBUG] _list_json_files called with base_path='{base_path}'")

        abs_base = os.path.abspath(base_path)

        if os.path.exists(base_path) and os.path.isdir(base_path):
            try:
                items = os.listdir(base_path)
                for item in items:
                    filepath = os.path.join(base_path, item)
                    if os.path.isfile(filepath) and item.endswith(".json"):
                        stat = os.stat(filepath)
                        files.append(
                            {
                                "path": filepath,
                                "name": item,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "relative_path": os.path.relpath(
                                    filepath, self.root_directory
                                ),
                            }
                        )
                        print(f"[DEBUG] Added JSON file: {item}")
            except PermissionError as e:
                print(f"[DEBUG] Files: PermissionError: {e}")

        files.sort(key=lambda x: x["mtime"], reverse=True)
        print(f"[DEBUG] Returning {len(files)} JSON files")
        return files

    def _list_directories(self, base_path=""):
        """List subdirectories in the specified directory."""
        dirs = []
        if not base_path:
            base_path = self.search_directory

        print(f"[DEBUG] _list_directories called with base_path='{base_path}'")
        print(f"[DEBUG] search_directory='{self.search_directory}'")
        print(f"[DEBUG] root_directory='{self.root_directory}'")

        abs_base = os.path.abspath(base_path)
        print(f"[DEBUG] abs_base='{abs_base}'")

        if os.path.exists(base_path) and os.path.isdir(base_path):
            try:
                items = os.listdir(base_path)
                print(f"[DEBUG] Found {len(items)} items in directory")
                for item in items:
                    dirpath = os.path.join(base_path, item)
                    if os.path.isdir(dirpath) and not item.startswith("."):
                        dirs.append(
                            {
                                "path": dirpath,
                                "name": item,
                                "relative_path": os.path.relpath(
                                    dirpath, self.root_directory
                                ),
                            }
                        )
                        print(f"[DEBUG] Added directory: {item}")
            except PermissionError as e:
                print(f"[DEBUG] PermissionError: {e}")
        else:
            print(f"[DEBUG] Path does not exist or is not a directory: {base_path}")

        dirs.sort(key=lambda x: x["name"])

        # Add parent directory option (always show .. to allow going up)
        parent = os.path.dirname(abs_base)
        if parent != abs_base:  # Not at filesystem root
            dirs.insert(
                0,
                {
                    "path": parent,
                    "name": "..",
                    "relative_path": os.path.relpath(parent, self.root_directory)
                    if parent != self.root_directory
                    else ".",
                },
            )
            print(f"[DEBUG] Added parent directory: {parent}")

        print(f"[DEBUG] Returning {len(dirs)} directories")
        return dirs


def start_server(
    port=8000, search_dir=None, initial_data=None, static_dir=None
):
    """Start the HTTP server."""
    ProfileVisualizerHandler.profiling_data = initial_data
    ProfileVisualizerHandler.search_directory = search_dir or os.getcwd()
    ProfileVisualizerHandler.root_directory = search_dir or os.getcwd()
    ProfileVisualizerHandler.static_dir = static_dir

    server = HTTPServer(("localhost", port), ProfileVisualizerHandler)

    print(f"\n{'='*60}")
    print("🔥 Profiling Visualizer Started!")
    print(f"{'='*60}")
    print(f"\nCurrent working directory: {os.getcwd()}")
    if initial_data:
        print(f"Initially loaded data: Yes")
        print(f"Timestamp: {initial_data.get('timestamp', 'N/A')}")
    print(f"\nSearch directory: {ProfileVisualizerHandler.search_directory}")
    print(f"\n👉 Open in browser: http://localhost:{port}")
    print("\nYou can:")
    print(
        f"  • Select files from the dropdown (scans: {ProfileVisualizerHandler.search_directory})"
    )
    print("  • Drag & drop JSON files into the browser")
    print("  • Click the drop zone to browse files")
    print("\nPress Ctrl+C to stop the server")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
