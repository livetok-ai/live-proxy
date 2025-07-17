import argparse
import asyncio
import logging
import os
import ssl
import time
from dataclasses import dataclass
from typing import Optional
import json

import aiohttp
from aiohttp import web

from connection import Connection
from logger import log_info
import metrics


@dataclass(frozen=True)
class ConnectionInfo:
    connection: Connection
    callback_url: Optional[str] = None
    metadata: Optional[dict] = None

    def __hash__(self):
        return self.connection.__hash__()


connections = set()  # Set of ConnectionInfo objects


async def _make_callback(connection, duration, callback_url, metadata):
    """Make callback request to notify about session closure"""
    try:
        async with aiohttp.ClientSession() as session:
            callback_data = {
                "session_id": connection.pc_id,
                "event": "session_closed",
                "timestamp": int(time.time() * 1000),
                "duration": int(duration * 1000),
                "transcript": connection.transcript,
            }
            # Add metadata if it was provided
            if metadata is not None:
                callback_data["metadata"] = metadata
            async with session.post(
                callback_url,
                json=callback_data,
                headers={"Content-Type": "application/json"},
            ) as response:
                log_info(
                    "Callback sent to %s, status: %d",
                    callback_url,
                    response.status,
                )
    except Exception as e:
        log_info("Failed to send callback to %s: %s", callback_url, e)


async def offer(request):
    def on_connection_closed(connection):
        # Find and remove the connection info
        conn_info = None
        for info in connections:
            if info.connection == connection:
                conn_info = info
                break

        if conn_info:
            connections.discard(conn_info)

            # Calculate duration and update metrics
            duration = time.time() - connection.start_time if connection.start_time else 0
            metrics.add_connection_duration(duration)
            metrics.set_open_connections(len(connections))

            # Make callback request if URL is provided
            if conn_info.callback_url:
                asyncio.create_task(_make_callback(connection, duration, conn_info.callback_url, conn_info.metadata))

    # Parse request
    content_type = request.headers.get("content-type", "").lower()

    if content_type == "application/json":
        # Parse JSON body with sdp, systemInstructions, callback, and metadata
        body = await request.json()
        sdp = body.get("sdp")
        system_instructions = body.get("system_instructions")
        callback_url = body.get("callback")
        metadata = body.get("metadata")
        if not sdp:
            raise web.HTTPBadRequest(text="Missing 'sdp' parameter in JSON body")
    else:
        # Backward compatibility: assume body is the SDP
        sdp = await request.text()
        system_instructions = None
        callback_url = None
        metadata = None

    model = request.query.get("model")

    # Create and start connection
    connection = Connection(on_closed=on_connection_closed)
    conn_info = ConnectionInfo(connection=connection, callback_url=callback_url, metadata=metadata)
    connections.add(conn_info)

    # Update metrics
    metrics.increment_connection()
    metrics.set_open_connections(len(connections))

    try:
        sdp_response = await connection.start(sdp, model, system_instructions)
        return web.Response(
            content_type="application/sdp",
            text=sdp_response,
        )
    except Exception as e:
        connections.discard(conn_info)
        metrics.set_open_connections(len(connections))
        raise


async def metrics_endpoint(request):
    """Prometheus metrics endpoint."""
    return web.Response(
        body=metrics.get_metrics(),
        content_type="text/plain; version=0.0.4",
        charset="utf-8",
    )


async def on_shutdown(app):
    # close peer connections
    coros = [conn_info.connection.close() for conn_info in connections]
    await asyncio.gather(*coros)
    connections.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Time LLM Proxy")
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("aioice").setLevel(level=logging.WARN)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/", offer)
    app.router.add_get("/metrics", metrics_endpoint)
    app.router.add_static("/demo", "demo")

    asyncio.run(
        web._run_app(
            app,
            access_log=None,
            host=args.host,
            port=args.port,
            ssl_context=ssl_context,
        )
    )
