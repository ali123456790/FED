"""
API server for FIDS server monitoring and management.

This module provides a REST API to monitor and manage the FIDS server.
It allows checking server status, viewing metrics, and controlling the server.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fids.server import FlowerServer
from fids.server.utils import (
    monitor_resources, get_available_snapshots, restore_server_state_from_snapshot,
    create_server_state_snapshot, generate_summary_report, export_model_for_deployment
)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FIDS Server API",
    description="API for monitoring and managing the FIDS federated learning server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server instance (will be set when the app starts)
server_instance: Optional[FlowerServer] = None
metrics_dir: str = os.environ.get("METRICS_DIR", "./metrics")

# Models for request/response validation
class ServerStatusResponse(BaseModel):
    status: str
    current_round: int = 0
    clients_connected: int = 0
    started_at: float = 0
    uptime_seconds: float = 0
    resources: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)

class MetricsResponse(BaseModel):
    summary: Dict[str, Any] = Field(default_factory=dict)
    rounds: List[Dict[str, Any]] = Field(default_factory=list)
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    clients: Dict[str, Any] = Field(default_factory=dict)

class SnapshotInfo(BaseModel):
    timestamp: int
    datetime: str
    round: int
    model_available: bool
    metrics_count: int

class SnapshotsResponse(BaseModel):
    snapshots: List[SnapshotInfo] = Field(default_factory=list)
    latest: Optional[SnapshotInfo] = None

class RestoreSnapshotRequest(BaseModel):
    timestamp: int

class ExportModelRequest(BaseModel):
    format: str = "saved_model"
    name: str = "fids_model"

class ExportModelResponse(BaseModel):
    success: bool
    path: str
    format: str
    timestamp: float

# Dependency for checking if server is initialized
def get_server():
    if server_instance is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return server_instance

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint for the API.
    
    Returns:
        Dictionary with health status
    """
    if server_instance is None:
        return {"status": "api_only"}
    
    return server_instance.health_check()

@app.get("/status", response_model=ServerStatusResponse)
async def get_server_status(server: FlowerServer = Depends(get_server)):
    """
    Get current server status.
    
    Returns:
        Server status information
    """
    # Get resource usage
    resources = monitor_resources()
    
    # Get server status
    status = "running"
    if hasattr(server, "server_thread") and not server.server_thread.is_alive():
        status = "stopped"
    
    # Calculate uptime
    started_at = getattr(server, "started_at", time.time())
    uptime = time.time() - started_at
    
    # Get client count
    clients_connected = 0
    if hasattr(server, "server") and hasattr(server.server, "client_manager"):
        clients_connected = len(server.server.client_manager.all())
    
    # Get current round
    current_round = getattr(server, "current_round", 0)
    
    return {
        "status": status,
        "current_round": current_round,
        "clients_connected": clients_connected,
        "started_at": started_at,
        "uptime_seconds": uptime,
        "resources": resources,
        "config": server.config
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(server: FlowerServer = Depends(get_server)):
    """
    Get server metrics.
    
    Returns:
        Dictionary with server metrics
    """
    # Generate summary report
    summary = generate_summary_report(metrics_dir)
    
    # Get last 10 round metrics
    rounds = getattr(server, "round_metrics", [])[-10:]
    
    # Get last 10 resource measurements
    resources = getattr(server, "resource_usage", [])[-10:]
    
    # Get client information
    clients = {}
    if hasattr(server, "server") and hasattr(server.server, "client_manager"):
        for client_id, client in server.server.client_manager.all().items():
            clients[client_id] = {
                "is_available": client.is_available(),
                "last_seen": getattr(client, "last_seen", 0)
            }
    
    return {
        "summary": summary,
        "rounds": rounds,
        "resources": resources,
        "clients": clients
    }

@app.post("/control/start")
async def start_server(background_tasks: BackgroundTasks, server: FlowerServer = Depends(get_server)):
    """
    Start the federated learning server if stopped.
    
    Returns:
        Status message
    """
    # Check if server is already running
    if hasattr(server, "server_thread") and server.server_thread.is_alive():
        return {"status": "already_running"}
    
    # Start server in background
    background_tasks.add_task(server.start, blocking=False)
    
    return {"status": "starting"}

@app.post("/control/stop")
async def stop_server(server: FlowerServer = Depends(get_server)):
    """
    Stop the federated learning server if running.
    
    Returns:
        Status message
    """
    # Check if server is running
    if not hasattr(server, "server_thread") or not server.server_thread.is_alive():
        return {"status": "not_running"}
    
    # Stop server
    server.stop()
    
    return {"status": "stopping"}

@app.post("/control/restart")
async def restart_server(background_tasks: BackgroundTasks, server: FlowerServer = Depends(get_server)):
    """
    Restart the federated learning server.
    
    Returns:
        Status message
    """
    # Stop server if running
    if hasattr(server, "server_thread") and server.server_thread.is_alive():
        server.stop()
    
    # Start server in background
    background_tasks.add_task(server.start, blocking=False)
    
    return {"status": "restarting"}

@app.post("/snapshots/create")
async def create_snapshot(server: FlowerServer = Depends(get_server)):
    """
    Create a server state snapshot.
    
    Returns:
        Status message and snapshot information
    """
    # Create snapshot
    create_server_state_snapshot(server, metrics_dir)
    
    # Get snapshot info
    snapshots = get_available_snapshots(metrics_dir)
    
    return {
        "status": "success",
        "snapshot": snapshots[0] if snapshots else None
    }

@app.get("/snapshots", response_model=SnapshotsResponse)
async def get_snapshots():
    """
    Get list of available snapshots.
    
    Returns:
        List of snapshot information
    """
    # Get snapshots
    snapshots = get_available_snapshots(metrics_dir)
    
    return {
        "snapshots": snapshots,
        "latest": snapshots[0] if snapshots else None
    }

@app.post("/snapshots/restore")
async def restore_snapshot(
    request: RestoreSnapshotRequest,
    server: FlowerServer = Depends(get_server)
):
    """
    Restore server state from a snapshot.
    
    Args:
        request: Request with snapshot timestamp
        
    Returns:
        Status message
    """
    # Restore snapshot
    success = restore_server_state_from_snapshot(server, metrics_dir, request.timestamp)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to restore snapshot")
    
    return {"status": "success"}

@app.post("/model/export", response_model=ExportModelResponse)
async def export_model(
    request: ExportModelRequest,
    server: FlowerServer = Depends(get_server)
):
    """
    Export model for deployment.
    
    Args:
        request: Export request with format and name
        
    Returns:
        Export status and path
    """
    # Check if model is available
    if not hasattr(server, "model") or server.model is None:
        raise HTTPException(status_code=400, detail="No model available")
    
    # Export directory
    export_dir = os.path.join(metrics_dir, "exported_models")
    
    # Export model
    path = export_model_for_deployment(
        server.model,
        export_dir,
        model_format=request.format,
        model_name=request.name
    )
    
    if not path:
        raise HTTPException(status_code=400, detail="Failed to export model")
    
    return {
        "success": True,
        "path": path,
        "format": request.format,
        "timestamp": time.time()
    }

@app.get("/model/download/{format}/{name}")
async def download_model(format: str, name: str):
    """
    Download an exported model.
    
    Args:
        format: Model format
        name: Model name
        
    Returns:
        File response with the model
    """
    # Determine file extension
    extensions = {
        "saved_model": "",
        "h5": ".h5",
        "tflite": ".tflite",
        "onnx": ".onnx",
        "pkl": ".pkl"
    }
    
    extension = extensions.get(format, "")
    
    # Build path
    if format == "saved_model":
        # For directory-based SavedModel
        model_path = os.path.join(metrics_dir, "exported_models", name)
        if not os.path.isdir(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Create zip file
        import shutil
        zip_path = os.path.join(metrics_dir, "exported_models", f"{name}.zip")
        shutil.make_archive(zip_path[:-4], 'zip', model_path)
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{name}.zip"
        )
    else:
        # For file-based models
        model_path = os.path.join(metrics_dir, "exported_models", f"{name}{extension}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        return FileResponse(
            model_path,
            filename=f"{name}{extension}"
        )

@app.get("/model/latest")
async def get_latest_model(server: FlowerServer = Depends(get_server)):
    """
    Get information about the latest model.
    
    Returns:
        Model information
    """
    # Check if model is available
    if not hasattr(server, "model") or server.model is None:
        raise HTTPException(status_code=404, detail="No model available")
    
    # Get model info
    model_info = {
        "type": server.model_config["type"],
        "name": server.model_config["name"],
        "timestamp": time.time()
    }
    
    # Get latest round with model
    if hasattr(server, "round_metrics") and server.round_metrics:
        latest_round = max(
            (m for m in server.round_metrics if m["phase"] == "fit"),
            key=lambda x: x["round"],
            default=None
        )
        if latest_round:
            model_info["round"] = latest_round["round"]
    
    return model_info

@app.get("/logs/{count}")
async def get_logs(count: int = 100):
    """
    Get recent log entries.
    
    Args:
        count: Number of recent log entries to retrieve
        
    Returns:
        List of log entries
    """
    # Check if log file exists
    log_file = os.environ.get("LOG_FILE", "./logs/fids.log")
    if not os.path.exists(log_file):
        return {"logs": []}
    
    # Read last N lines
    try:
        from collections import deque
        with open(log_file, 'r') as f:
            return {"logs": list(deque(f, count))}
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return {"logs": [f"Error reading log file: {str(e)}"]}

def init_api_server(server: FlowerServer):
    """
    Initialize the API server with a FlowerServer instance.
    
    Args:
        server: The FlowerServer instance to use
    """
    global server_instance
    server_instance = server

def run_api_server(host: str = "0.0.0.0", port: int = 8081):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Run standalone API server (without FlowerServer)
    import argparse
    
    parser = argparse.ArgumentParser(description="FIDS Server API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    parser.add_argument("--metrics", type=str, default="./metrics", help="Metrics directory")
    args = parser.parse_args()
    
    # Set metrics directory
    metrics_dir = args.metrics
    
    # Run API server
    run_api_server(args.host, args.port)