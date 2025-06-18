"""
Base Backend Agent for Symphony/Pie model backends.

This module provides a base class that handles:
- Registration with the engine-management-service
- Heartbeat maintenance
- Local management API for receiving commands
- Model loading/unloading operations

Model-specific backends should inherit from BaseBackendAgent and implement
the model-specific logic.
"""

import requests
import threading
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import socket
import logging
import os
import signal
import sys
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoadRequest(BaseModel):
    model_name: str
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = {}

class ModelUnloadRequest(BaseModel):
    model_name: str

class BaseBackendAgent(ABC):
    """
    Base class for Symphony/Pie backend agents.

    Handles registration, heartbeat, and basic management API.
    Model-specific backends should inherit and implement abstract methods.
    """

    def __init__(self,
                 management_service_url: str,
                 backend_host: str,
                 backend_api_port: int,
                 backend_type: str,
                 service_name: str,
                 ipc_endpoint: str = None,
                 supported_models: list = None,
                 shutdown_callback: Callable = None):
        """
        Initialize the backend agent.

        Args:
            management_service_url: URL of the engine management service
            backend_host: Host for the backend's local management API
            backend_api_port: Port for the backend's local management API
            backend_type: Type identifier for this backend (e.g., "qwen3", "llama3")
            service_name: Human-readable name for this service
            ipc_endpoint: ZMQ IPC endpoint for data plane communication
            supported_models: List of model names this backend can handle
            shutdown_callback: Function to call when termination is requested
        """
        self.management_service_url = management_service_url.rstrip('/')
        self.backend_host = backend_host
        self.backend_api_port = int(backend_api_port)
        self.backend_type = backend_type
        self.service_name = service_name
        self.ipc_endpoint = ipc_endpoint
        self.shutdown_callback = shutdown_callback

        self.capabilities = {
            "type": backend_type,
            "name": service_name,
            "supported_models": supported_models or [],
            "ipc_endpoint": ipc_endpoint
        }

        # Backend ID will be assigned by management service on registration
        self.backend_id = None

        self.heartbeat_thread = None
        self.api_thread = None
        self.stop_event = threading.Event()

        # Track loaded models
        self.loaded_models = {}

        # Setup FastAPI app
        self.local_api_app = FastAPI(
            title=f"{service_name} Management API",
            version="0.1.0",
            description=f"Management API for {service_name} backend"
        )
        self._setup_routes()

    def _setup_routes(self):
        """Setup the management API routes."""

        @self.local_api_app.get("/manage/health", tags=["Management"])
        async def health_check():
            """Provides the health status of the backend agent."""
            return {
                "status": "healthy",
                "backend_id": self.backend_id,
                "service_name": self.service_name,
                "backend_type": self.backend_type,
                "loaded_models": list(self.loaded_models.keys()),
                "ipc_endpoint": self.ipc_endpoint
            }

        @self.local_api_app.post("/manage/terminate", tags=["Management"])
        async def terminate_backend():
            """Signals the backend agent and main service to terminate."""
            logger.info(f"Received terminate command for backend {self.backend_id} ({self.service_name}). Initiating shutdown.")

            # Call shutdown callback if provided
            if self.shutdown_callback:
                try:
                    self.shutdown_callback()
                except Exception as e:
                    logger.error(f"Error in shutdown callback: {e}")

            # Signal agent to stop
            self.stop()

            # Schedule process termination after a brief delay to allow response
            def delayed_exit():
                time.sleep(1)
                os._exit(0)

            threading.Thread(target=delayed_exit, daemon=True).start()

            return {"message": "Termination initiated. Process will exit shortly."}

        @self.local_api_app.post("/manage/models/load", tags=["Model Management"])
        async def load_model(request: ModelLoadRequest):
            """Load a model into the backend."""
            try:
                logger.info(f"Loading model: {request.model_name}")
                result = await self._load_model_impl(request)

                if result.get("success", False):
                    self.loaded_models[request.model_name] = {
                        "model_path": request.model_path,
                        "model_type": request.model_type,
                        "load_time": time.time(),
                        "additional_params": request.additional_params
                    }
                    logger.info(f"Successfully loaded model: {request.model_name}")

                return result

            except Exception as e:
                logger.error(f"Failed to load model {request.model_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

        @self.local_api_app.post("/manage/models/unload", tags=["Model Management"])
        async def unload_model(request: ModelUnloadRequest):
            """Unload a model from the backend."""
            try:
                logger.info(f"Unloading model: {request.model_name}")
                result = await self._unload_model_impl(request)

                if result.get("success", False):
                    self.loaded_models.pop(request.model_name, None)
                    logger.info(f"Successfully unloaded model: {request.model_name}")

                return result

            except Exception as e:
                logger.error(f"Failed to unload model {request.model_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

        @self.local_api_app.get("/manage/models", tags=["Model Management"])
        async def list_loaded_models():
            """List currently loaded models."""
            return {
                "loaded_models": self.loaded_models,
                "supported_models": self.capabilities.get("supported_models", [])
            }

    @abstractmethod
    async def _load_model_impl(self, request: ModelLoadRequest) -> Dict[str, Any]:
        """
        Implementation-specific model loading logic.

        Args:
            request: Model load request with model details

        Returns:
            Dict with success status and any additional info
        """
        pass

    @abstractmethod
    async def _unload_model_impl(self, request: ModelUnloadRequest) -> Dict[str, Any]:
        """
        Implementation-specific model unloading logic.

        Args:
            request: Model unload request

        Returns:
            Dict with success status and any additional info
        """
        pass

    def get_management_api_address(self):
        """Get the address to report for this backend's management API."""
        host_to_report = self.backend_host
        if host_to_report == "0.0.0.0" or host_to_report == "::":
            try:
                # Try to get a resolvable hostname or IP
                hostname = socket.getfqdn()
                if hostname and not ("localhost" in hostname or ".local" in hostname):
                    host_to_report = hostname
                else:
                    # Get IP address by connecting to a public DNS server
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(0.1)
                    s.connect(("8.8.8.8", 80))
                    ip_address = s.getsockname()[0]
                    s.close()
                    if ip_address != "0.0.0.0":
                        host_to_report = ip_address
                    else:
                        host_to_report = "127.0.0.1"
                logger.info(f"Determined host_to_report as: {host_to_report}")
            except socket.error as e:
                logger.warning(f"Could not determine IP/hostname, using '127.0.0.1'. Error: {e}")
                host_to_report = "127.0.0.1"

        return f"http://{host_to_report}:{self.backend_api_port}"

    def register(self):
        """Register this backend with the engine management service."""
        if self.backend_id:
            logger.info(f"Backend ID {self.backend_id} already exists. Verifying with management service.")
            return True

        registration_url = f"{self.management_service_url}/backends/register"

        # Convert capabilities to the format expected by engine-management-service
        # Capabilities should be a list of strings, not an object
        capabilities_list = []
        if isinstance(self.capabilities, dict):
            # Convert our dict-based capabilities to a list of strings
            capabilities_list.append(f"type:{self.capabilities.get('type', 'unknown')}")
            capabilities_list.append(f"name:{self.capabilities.get('name', 'unknown')}")
            if self.capabilities.get('supported_models'):
                for model in self.capabilities['supported_models']:
                    capabilities_list.append(f"model:{model}")
            if self.capabilities.get('ipc_endpoint'):
                capabilities_list.append(f"ipc_endpoint:{self.capabilities['ipc_endpoint']}")
        else:
            capabilities_list = self.capabilities if self.capabilities else []

        payload = {
            "capabilities": capabilities_list,
            "management_api_address": self.get_management_api_address()
        }

        try:
            logger.info(f"Registering backend '{self.service_name}' to {registration_url}")
            response = requests.post(registration_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.backend_id = data.get("backend_id")

            if self.backend_id:
                logger.info(f"Backend registered successfully. Received backend_id: {self.backend_id}")
                return True
            else:
                logger.error(f"Registration response did not include backend_id. Response: {data}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register backend '{self.service_name}': {e}")
            return False

    def _send_heartbeat_loop(self):
        """Main heartbeat loop."""
        heartbeat_url_template = f"{self.management_service_url}/backends/{{backend_id}}/heartbeat"

        # Initial registration attempt if we don't have a backend_id
        if not self.backend_id:
            logger.info("No backend_id found, attempting initial registration.")
            if not self.register():
                logger.warning("Initial registration failed. Will retry periodically during heartbeat loop.")

        while not self.stop_event.is_set():
            # If we don't have a backend_id, try to register
            if not self.backend_id:
                logger.info("Backend ID is missing, attempting registration.")
                if not self.register():
                    logger.warning("Registration failed. Will retry in 30 seconds.")
                    self.stop_event.wait(30)
                    continue  # Skip heartbeat attempt and try registration again

            # We have a backend_id, try to send heartbeat
            current_heartbeat_url = heartbeat_url_template.format(backend_id=self.backend_id)
            try:
                logger.debug(f"Sending heartbeat for '{self.service_name}' to {current_heartbeat_url}")
                response = requests.post(current_heartbeat_url, timeout=5)
                response.raise_for_status()
                logger.debug(f"Heartbeat successful for '{self.service_name}'. Status: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send heartbeat for '{self.service_name}': {e}")
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    logger.warning(f"Heartbeat failed with 404 for backend_id {self.backend_id}. "
                                   "Management service may have restarted. Clearing backend_id.")
                    self.backend_id = None
                    # Will attempt re-registration on next loop iteration

            self.stop_event.wait(10)  # Wait 10 seconds between heartbeats

        logger.info(f"Heartbeat loop stopped for '{self.service_name}'.")

    def start_heartbeat(self):
        """Start the heartbeat thread."""
        self.heartbeat_thread = threading.Thread(target=self._send_heartbeat_loop, daemon=True)
        self.heartbeat_thread.name = f"{self.service_name}-HeartbeatThread"
        self.heartbeat_thread.start()
        logger.info(f"Heartbeat thread started for '{self.service_name}'.")

    def start_local_management_api(self):
        """Start the local management API server."""
        def run_api():
            try:
                logger.info(f"Starting local management API for '{self.service_name}' on {self.backend_host}:{self.backend_api_port}")
                uvicorn.run(
                    self.local_api_app,
                    host=self.backend_host,
                    port=self.backend_api_port,
                    log_level="warning",
                    access_log=False  # Reduce log noise
                )
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    logger.error(f"Port {self.backend_api_port} is already in use. Please use a different port.")
                else:
                    logger.error(f"Failed to start local management API for '{self.service_name}': {e}")
                self.stop_event.set()
            except Exception as e:
                logger.error(f"Failed to start local management API for '{self.service_name}': {e}", exc_info=True)
                self.stop_event.set()

        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.name = f"{self.service_name}-ApiThread"
        self.api_thread.start()
        logger.info(f"Local management API thread started for '{self.service_name}'.")

    def start(self):
        """Start both heartbeat and management API."""
        # Start heartbeat (which handles registration) and management API
        # No need for initial registration here - heartbeat loop handles it
        self.start_heartbeat()
        self.start_local_management_api()
        return True

    def stop(self):
        """Stop the backend agent."""
        logger.info(f"Stopping BackendAgent for '{self.service_name}'...")
        self.stop_event.set()

        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            logger.debug(f"Waiting for heartbeat thread of '{self.service_name}' to join...")
            self.heartbeat_thread.join(timeout=5)
            if self.heartbeat_thread.is_alive():
                logger.warning(f"Heartbeat thread of '{self.service_name}' did not join in time.")

        logger.info(f"BackendAgent for '{self.service_name}' stop sequence complete.")

    def wait_until_stopped(self):
        """Wait until the stop event is set."""
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"KeyboardInterrupt received, stopping agent '{self.service_name}'...")
            self.stop()
