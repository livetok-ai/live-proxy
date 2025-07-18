import argparse
import asyncio
import logging
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
    callback: Optional[str] = None
    metadata: Optional[dict] = None

    def __hash__(self):
        return self.connection.__hash__()


connections = set()  # Set of ConnectionInfo objects


async def _make_callback(connection, duration, callback, metadata):
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "connection_id": connection.id,
                "event": "session_closed",
                "timestamp": int(time.time() * 1000),
                "duration": int(duration * 1000),
                "transcript": connection.transcript,
                "metadata": metadata,
            }
            async with session.post(
                callback,
                json=data,
                headers={"Content-Type": "application/json"},
            ) as response:
                log_info(
                    "Callback sent to %s, status: %d",
                    callback,
                    response.status,
                )
    except Exception as e:
        log_info("Failed to send callback to %s: %s", callback, e)


async def create_connection(request):
    def on_connection_closed(connection):
        # Find and remove the connection info
        conn_info = None
        for info in connections:
            if info.connection == connection:
                conn_info = info
                break

        log_info(f"Connection closed: {conn_info}")

        if conn_info:
            connections.discard(conn_info)

            # Calculate duration and update metrics
            duration = time.time() - connection.start_time if connection.start_time else 0
            metrics.add_connection_duration(duration)
            metrics.set_open_connections(len(connections))

            # Make callback request if URL is provided
            if conn_info.callback:
                asyncio.create_task(_make_callback(connection, duration, conn_info.callback, conn_info.metadata))

    body = await request.json()
    sdp = body.get("sdp")
    system_instructions = body.get("system_instructions")
    callback = body.get("callback")
    metadata = body.get("metadata")
    if not sdp:
        raise web.HTTPBadRequest(text="Missing 'sdp' parameter in JSON body")

    model = request.query.get("model")

    # Create and start connection
    connection = Connection(on_closed=on_connection_closed)
    conn_info = ConnectionInfo(connection=connection, callback=callback, metadata=metadata)
    connections.add(conn_info)

    # Update metrics
    metrics.increment_connection()
    metrics.set_open_connections(len(connections))

    try:
        sdp_response = await connection.start(sdp, model, system_instructions)
        return web.Response(
            content_type="application/json",
            body=json.dumps(
                {
                    "sdp": sdp_response,
                    "id": conn_info.connection.id,
                }
            ),
        )
    except Exception as e:
        connections.discard(conn_info)
        metrics.set_open_connections(len(connections))
        raise


async def delete_connection(request):
    connection_id = request.match_info.get("connection_id")

    if not connection_id:
        raise web.HTTPBadRequest(text="Missing connection_id")

    conn_info = None
    for info in connections:
        if info.connection.id == connection_id:
            conn_info = info
            break

    if not conn_info:
        raise web.HTTPNotFound(text="Connection not found")

    try:
        await conn_info.connection.close()
        return web.Response(
            status=200, body=json.dumps({"message": "Connection closed successfully"}), content_type="application/json"
        )
    except Exception as e:
        log_info(f"Error closing connection {connection_id}: {e}")
        raise web.HTTPInternalServerError(text="Failed to close connection")


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
    app.router.add_post("/connection", create_connection)
    app.router.add_delete("/connection/{connection_id}", delete_connection)
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
