import argparse
import asyncio
import json
import logging
import ssl
import sys
import time

import aiohttp
from aiohttp import web
from aiohttp_cors import ResourceOptions
from aiohttp_cors import setup as setup_cors
from dotenv import load_dotenv

import metrics
from connection import ConnectionManager
from logger import log_info
from sip import SIPServer

# Load environment variables from .env file
load_dotenv()


def in_venv():
    return sys.prefix != sys.base_prefix


connections = ConnectionManager()


def _on_connection_closed(conn_info):
    """Callback invoked when a connection is closed."""
    log_info(f"Connection closed: {conn_info.metadata if conn_info.metadata else 'Unknown'}")

    # Calculate duration and update metrics
    duration = time.time() - conn_info.connection.start_time if conn_info.connection.start_time else 0
    metrics.add_connection_duration(duration)

    connections.remove_connection(conn_info)

    # Make callback request if URL is provided
    if conn_info.callback:
        asyncio.create_task(_closed_request(conn_info.connection, duration, conn_info.callback, conn_info.metadata))


async def _on_tool_call(conn_info, tool_name, tool_id, parameters, tools):
    """Callback invoked when a tool is called."""
    return await _tool_call_request(conn_info.connection, tool_name, tool_id, parameters, tools, conn_info.metadata)


# Configure ConnectionManager callbacks
connections.configure_callbacks(closed_callback=_on_connection_closed, tool_call_callback=_on_tool_call)


def _parse_request_body(body):
    """Parse and extract common parameters from request body."""
    keys = ["sdp", "system_instructions", "callback", "tools", "metadata", "voice", "language", "model", "api_key"]
    return {key: body.get(key) for key in keys}


def _get_connection_or_raise(connection_id):
    """Find a connection by ID or raise appropriate HTTP error."""
    if not connection_id:
        raise web.HTTPBadRequest(text="Missing connection_id")

    conn_info = connections.find_connection_by_id(connection_id)
    if not conn_info:
        raise web.HTTPNotFound(text="Connection not found")

    return conn_info


async def _closed_request(connection, duration, callback, metadata):
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


async def _tool_call_request(connection, tool_name, tool_id, parameters, tools, metadata):
    """Make HTTP request to tool and return response"""
    tool = next((t for t in tools if t["name"] == tool_name), None)
    if not tool:
        log_info(f"Tool call: {tool_name} not found", context=connection.id)
        return {"error": f"Tool {tool_name} not found"}

    log_info(f"Tool call: {tool_name} found", context=connection.id)

    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "parameters": parameters,
                "id": tool_id,
                "name": tool_name,
                "connection_id": connection.id,
                "timestamp": int(time.time() * 1000),
                "metadata": metadata,
            }
            async with session.post(
                tool["url"],
                json=data,
                headers={"Content-Type": "application/json"},
            ) as response:
                log_info(
                    "Tool request sent to %s, status: %d",
                    tool["url"],
                    response.status,
                    context=connection.id,
                )
                return await response.json()
    except Exception as e:
        log_info(f"Error calling tool {tool_name}: {e}", context=connection.id)
        return {"error": str(e)}


async def create_connection(request):
    log_info("HTTP create connection request")

    body = await request.json()
    params = _parse_request_body(body)

    if not params["sdp"]:
        raise web.HTTPBadRequest(text="Missing 'sdp' parameter in JSON body")

    log_info(
        f"Creating connection model: {params['model']} callback: {params['callback']} instructions: {params['system_instructions'][:100] if params['system_instructions'] else None} metadata: {params['metadata']} voice: {params['voice']} language: {params['language']} tools: {params['tools']}"
    )

    # Create and start connection using ConnectionManager
    conn_info = connections.create_connection(callback=params["callback"], metadata=params["metadata"])

    try:
        sdp_response = await conn_info.connection.start(
            params["sdp"],
            params["model"],
            params["system_instructions"],
            params["tools"],
            params["voice"],
            params["language"],
            params["api_key"],
        )
        return web.Response(
            content_type="application/json",
            body=json.dumps(
                {
                    "sdp": sdp_response,
                    "id": conn_info.connection.id,
                }
            ),
        )
    except Exception:
        connections.remove_connection(conn_info)
        raise


async def delete_connection(request):
    connection_id = request.match_info.get("connection_id")

    log_info(f"HTTP delete connection request {connection_id}")

    conn_info = _get_connection_or_raise(connection_id)

    try:
        await conn_info.connection.close()
        return web.Response(
            status=200, body=json.dumps({"message": "Connection closed successfully"}), content_type="application/json"
        )
    except Exception as e:
        log_info(f"Error closing connection {connection_id}: {e}")
        raise web.HTTPInternalServerError(text="Failed to close connection") from e


async def update_connection(request):
    connection_id = request.match_info.get("connection_id")

    log_info(f"HTTP update connection request {connection_id}")

    conn_info = _get_connection_or_raise(connection_id)

    try:
        # await conn_info.connection.update()
        return web.Response(
            status=200, body=json.dumps({"message": "Connection updated successfully"}), content_type="application/json"
        )
    except Exception as e:
        log_info(f"Error updating connection {connection_id}: {e}")
        raise web.HTTPInternalServerError(text="Failed to update connection") from e


async def metrics_endpoint(request):
    """Prometheus metrics endpoint."""
    return web.Response(
        body=metrics.get_metrics(),
        content_type="text/plain; version=0.0.4",
        charset="utf-8",
    )


async def on_shutdown(app):
    # close peer connections
    await connections.close_all()
    # Stop SIP server if running
    if hasattr(app, "sip_server"):
        await app.sip_server.stop()


async def run_servers(app, host, port, ssl_context, sip_host, sip_port, sip_callback_url):
    """Run both the web app and SIP server concurrently."""
    # Create SIP server
    sip_server = SIPServer(host=sip_host, port=sip_port, callback_url=sip_callback_url)
    app.sip_server = sip_server

    # Create tasks for both servers
    log_info(f"HTTP server listening on {host}:{port}")
    web_task = asyncio.create_task(
        web._run_app(
            app,
            access_log=None,
            host=host,
            port=port,
            ssl_context=ssl_context,
        )
    )

    sip_task = asyncio.create_task(sip_server.start())

    # Wait for both to complete (or until one fails)
    try:
        await asyncio.gather(web_task, sip_task)
    except Exception as e:
        log_info(f"Error running servers: {e}")
        raise


if __name__ == "__main__":
    if not in_venv():
        log_info("RUNNING IN A NON VENV")

    parser = argparse.ArgumentParser(description="Real Time LLM Proxy")
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sip-host", default=None)
    parser.add_argument("--sip-port", type=int, default=5060)
    parser.add_argument("--sip-callback-url")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger("aioice").setLevel(level=logging.WARN)

    if args.cert_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    # Setup CORS
    cors = setup_cors(
        app,
        defaults={
            "*": ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")
        },
    )

    # Add routes with CORS
    cors.add(app.router.add_post("/connection", create_connection))
    cors.add(app.router.add_delete("/connection/{connection_id}", delete_connection))
    cors.add(app.router.add_put("/connection/{connection_id}", update_connection))
    cors.add(app.router.add_get("/metrics", metrics_endpoint))
    app.router.add_static("/demo", "demo")

    asyncio.run(
        run_servers(
            app,
            args.host,
            args.port,
            ssl_context,
            args.sip_host,
            args.sip_port,
            args.sip_callback_url,
        )
    )
