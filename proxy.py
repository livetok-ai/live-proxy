import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time

from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv()

import aiohttp
from aiohttp import web

import metrics


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.Response:
    if request.method == "OPTIONS":
        return web.Response(
            headers={
                "Access-Control-Allow-Origin": request.headers.get("Origin", "*"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "3600",
            }
        )
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response


from connection import ConnectionManager
from interfaces.rtmp import RTMPServer
from interfaces.sip import SIPServer
from interfaces.websocket import WebSocketServer
from interfaces.webtransport import WebTransportServer
from logger import log_info, log_warn
from session import SessionManager


def in_venv():
    return sys.prefix != sys.base_prefix


connections = ConnectionManager()
webtransport_server: "WebTransportServer | None" = None
websocket_server: "WebSocketServer | None" = None


class HTTPServer:
    def __init__(self, host, port, ssl_context):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.app = web.Application(middlewares=[cors_middleware])
        self._setup_routes()
        self.app.on_shutdown.append(self._on_shutdown)

    def _setup_routes(self):
        self.app.router.add_post("/connection", self.create_connection)
        self.app.router.add_delete("/connection/{connection_id}", self.delete_connection)
        self.app.router.add_put("/connection/{connection_id}", self.update_connection)
        self.app.router.add_get("/session", self.list_sessions)
        self.app.router.add_post("/session/{session_id}/connection", self.create_session_connection)
        self.app.router.add_get("/metrics", self.metrics_endpoint)
        self.app.router.add_get("/webtransport-info", self.webtransport_info)
        self.app.router.add_get("/websocket-info", self.websocket_info)
        self.app.router.add_static("/demo", os.path.join(os.path.dirname(__file__), "demo"))

    async def _on_shutdown(self, app):
        await connections.close_all()

    async def start(self):
        log_info(f"HTTP server listening on {self.host}:{self.port} (TCP)")
        await web._run_app(
            self.app,
            access_log=None,
            print=None,
            host=self.host,
            port=self.port,
            ssl_context=self.ssl_context,
        )

    def _parse_request_body(self, body):
        """Parse and extract common parameters from request body."""
        keys = [
            "sdp",
            "system_instructions",
            "callback",
            "tools",
            "metadata",
            "voice",
            "language",
            "model",
            "api_key",
            "script",
            "recording",
        ]
        return {key: body.get(key) for key in keys}

    def _get_connection_or_raise(self, connection_id):
        """Find a connection by ID or raise appropriate HTTP error."""
        if not connection_id:
            raise web.HTTPBadRequest(text="Missing connection_id")

        conn_info = connections.find_connection_by_id(connection_id)
        if not conn_info:
            raise web.HTTPNotFound(text="Connection not found")

        return conn_info

    async def create_connection(self, request):
        log_info("HTTP create connection request")

        body = await request.json()
        params = self._parse_request_body(body)

        if not params["sdp"]:
            raise web.HTTPBadRequest(text="Missing 'sdp' parameter in JSON body")

        log_info(
            f"Creating connection model: {params['model']} callback: {params['callback']} instructions: {params['system_instructions'][:100] if params['system_instructions'] else None} metadata: {params['metadata']} voice: {params['voice']} language: {params['language']} tools: {len(params['tools']) if params['tools'] else 0} script: {True if params['script'] else False}"
        )

        # Create and start connection using ConnectionManager
        conn_info = connections.create_connection(
            callback=params["callback"], metadata=params["metadata"], recording=params["recording"]
        )

        try:
            sdp_response = await conn_info.connection.start(
                params["sdp"],
                params["model"],
                params["system_instructions"],
                params["tools"],
                params["voice"],
                params["language"],
                params["api_key"],
                params["script"],
            )
            return web.Response(
                content_type="application/json",
                body=json.dumps(
                    {
                        "sdp": sdp_response,
                        "id": conn_info.connection.id,
                        "sessionId": conn_info.connection.session.id,
                    }
                ),
            )
        except web.HTTPException:
            connections.remove_connection(conn_info)
            raise
        except Exception as e:
            connections.remove_connection(conn_info)
            log_warn(f"Failed to start connection {conn_info.connection.id}: {e}")
            raise web.HTTPBadGateway(text=f"Failed to start connection: {e}") from None

    async def list_sessions(self, request):
        """List all running sessions and their attached connections."""
        sessions = SessionManager().list_sessions()
        return web.Response(
            content_type="application/json",
            body=json.dumps(
                {
                    "sessions": [
                        {
                            "id": session.id,
                            "created_at": int(session.created_at * 1000),
                            "connections": [connection.id for connection in session.connections],
                        }
                        for session in sessions
                    ]
                }
            ),
        )

    async def create_session_connection(self, request):
        """Connect to an existing session over WebRTC to watch and participate in it."""
        session_id = request.match_info.get("session_id")
        log_info(f"HTTP create session connection request {session_id}")

        session = SessionManager().find_session_by_id(session_id)
        if not session:
            raise web.HTTPNotFound(text="Session not found")

        body = await request.json()
        sdp = body.get("sdp")
        if not sdp:
            raise web.HTTPBadRequest(text="Missing 'sdp' parameter in JSON body")

        conn_info = connections.create_connection(
            callback=body.get("callback"), metadata=body.get("metadata"), session=session
        )

        try:
            sdp_response = await conn_info.connection.start_join(sdp)
            return web.Response(
                content_type="application/json",
                body=json.dumps(
                    {
                        "sdp": sdp_response,
                        "id": conn_info.connection.id,
                        "sessionId": session.id,
                    }
                ),
            )
        except web.HTTPException:
            connections.remove_connection(conn_info)
            raise
        except Exception as e:
            connections.remove_connection(conn_info)
            log_warn(f"Failed to join session {session.id}: {e}")
            raise web.HTTPBadGateway(text=f"Failed to join session: {e}") from None

    async def delete_connection(self, request):
        connection_id = request.match_info.get("connection_id")

        log_info(f"HTTP delete connection request {connection_id}")

        conn_info = self._get_connection_or_raise(connection_id)

        try:
            await conn_info.connection.close()
            return web.Response(
                status=200,
                body=json.dumps({"message": "Connection closed successfully"}),
                content_type="application/json",
            )
        except Exception as e:
            log_info(f"Error closing connection {connection_id}: {e}")
            raise web.HTTPInternalServerError(text="Failed to close connection") from e

    async def update_connection(self, request):
        connection_id = request.match_info.get("connection_id")

        log_info(f"HTTP update connection request {connection_id}")

        self._get_connection_or_raise(connection_id)

        try:
            # await conn_info.connection.update()
            return web.Response(
                status=200,
                body=json.dumps({"message": "Connection updated successfully"}),
                content_type="application/json",
            )
        except Exception as e:
            log_info(f"Error updating connection {connection_id}: {e}")
            raise web.HTTPInternalServerError(text="Failed to update connection") from e

    async def metrics_endpoint(self, request):
        """Prometheus metrics endpoint."""
        return web.Response(
            body=metrics.get_metrics(),
            content_type="text/plain; version=0.0.4",
            charset="utf-8",
        )

    async def webtransport_info(self, request):
        """Connection info for the WebTransport demo page (port + self-signed cert hash, if any)."""
        if not webtransport_server:
            raise web.HTTPNotFound(text="WebTransport server not running")
        return web.Response(
            content_type="application/json",
            body=json.dumps(
                {
                    "port": webtransport_server.port,
                    "certificateHash": webtransport_server.certificate_hash_b64,
                }
            ),
        )

    async def websocket_info(self, request):
        """Connection info for the WebSocket demo page (port + whether it's using TLS)."""
        if not websocket_server:
            raise web.HTTPNotFound(text="WebSocket server not running")
        return web.Response(
            content_type="application/json",
            body=json.dumps(
                {
                    "port": websocket_server.port,
                    "secure": websocket_server.secure,
                }
            ),
        )


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


async def _on_webtransport_session(pc, params):
    """Called once a WebTransport/raw-QUIC session sends its init control message
    (JSON with the same fields as the HTTP /connection body, minus 'sdp')."""
    log_info(f"WebTransport session starting model={params.get('model')} metadata={params.get('metadata')}")

    conn_info = connections.create_connection(callback=params.get("callback"), metadata=params.get("metadata"))
    try:
        await conn_info.connection.start_webtransport(
            pc,
            model=params.get("model"),
            system_instructions=params.get("system_instructions"),
            tools=params.get("tools"),
            voice=params.get("voice"),
            language=params.get("language"),
            api_key=params.get("api_key"),
            script=params.get("script"),
        )
    except Exception:
        connections.remove_connection(conn_info)
        raise


async def _on_websocket_session(pc, params):
    """Called once a WebSocket session sends its init control message
    (JSON with the same fields as the HTTP /connection body, minus 'sdp')."""
    log_info(f"WebSocket session starting model={params.get('model')} metadata={params.get('metadata')}")

    conn_info = connections.create_connection(callback=params.get("callback"), metadata=params.get("metadata"))
    try:
        await conn_info.connection.start_webtransport(
            pc,
            model=params.get("model"),
            system_instructions=params.get("system_instructions"),
            tools=params.get("tools"),
            voice=params.get("voice"),
            language=params.get("language"),
            api_key=params.get("api_key"),
            script=params.get("script"),
        )
    except Exception:
        connections.remove_connection(conn_info)
        raise


async def _on_rtmp_session(pc, params):
    """Called once an RTMP client starts publishing (params parsed from the stream key query string)."""
    log_info(f"RTMP session starting model={params.get('model')} metadata={params.get('metadata')}")

    conn_info = connections.create_connection(callback=params.get("callback"), metadata=params.get("metadata"))
    try:
        await conn_info.connection.start_rtmp(
            pc,
            model=params.get("model"),
            system_instructions=params.get("system_instructions"),
            tools=params.get("tools"),
            voice=params.get("voice"),
            language=params.get("language"),
            api_key=params.get("api_key"),
            script=params.get("script"),
        )
    except Exception:
        connections.remove_connection(conn_info)
        raise


async def setup_all_providers():
    """Eagerly run every provider's setup() (model downloads/loads) up front,
    so the first real connection isn't stuck waiting on it."""
    from connection import MODEL_MAP

    log_info("Initializing and setting up all models...")
    for model_cls in set(MODEL_MAP.values()):
        try:
            await model_cls.setup()
            log_info(f"Set up model class: {model_cls.__name__}")
        except Exception as e:
            log_info(f"Error setting up model class {model_cls.__name__}: {e}")


async def run_servers(
    host,
    port,
    ssl_context,
    sip_host,
    sip_port,
    sip_callback_url,
    wt_host,
    wt_port,
    wt_ssl_context,
    ws_host,
    ws_port,
    ws_ssl_context,
    rtmp_host,
    rtmp_port,
    rtmp_callback_url,
    setup=False,
):
    """Run the web app, SIP server, WebTransport/QUIC server, WebSocket server and RTMP server concurrently."""
    if setup:
        await setup_all_providers()

    global webtransport_server, websocket_server

    http_server = HTTPServer(host=host, port=port, ssl_context=ssl_context)
    sip_server = SIPServer(host=sip_host, port=sip_port, callback_url=sip_callback_url)
    webtransport_server = WebTransportServer(
        host=wt_host,
        port=wt_port,
        on_session=_on_webtransport_session,
        certificate=wt_ssl_context[0] if wt_ssl_context else None,
        private_key=wt_ssl_context[1] if wt_ssl_context else None,
    )
    websocket_server = WebSocketServer(
        host=ws_host,
        port=ws_port,
        on_session=_on_websocket_session,
        ssl_context=ws_ssl_context,
    )

    rtmp_server = RTMPServer(
        host=rtmp_host, port=rtmp_port, on_session=_on_rtmp_session, callback_url=rtmp_callback_url
    )

    # Bind all servers first so every "listening" log shows up together at startup
    try:
        await sip_server.bind()
        await webtransport_server.bind()
        await websocket_server.bind()
        await rtmp_server.bind()
        # Wait for all to complete (or until one fails)
        await asyncio.gather(
            http_server.start(),
            sip_server.start(),
            webtransport_server.start(),
            websocket_server.start(),
            rtmp_server.start(),
        )
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
    parser.add_argument("--sip-callback-url", default=os.getenv("SIP_CALLBACK_URL"))
    parser.add_argument("--wt-host", default="0.0.0.0")
    parser.add_argument("--wt-port", type=int, default=4433)
    parser.add_argument("--wt-cert-file")
    parser.add_argument("--wt-key-file")
    parser.add_argument("--ws-host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--ws-cert-file")
    parser.add_argument("--ws-key-file")
    parser.add_argument("--rtmp-host", default="0.0.0.0")
    parser.add_argument("--rtmp-port", type=int, default=1935)
    parser.add_argument("--rtmp-callback-url", default=os.getenv("RTMP_CALLBACK_URL"))
    parser.add_argument("--log-level", default="INFO", help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Eagerly load/set up all providers (download and load models) before starting the servers, "
        "instead of lazily on first use",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("aioice").setLevel(level=logging.WARN)
    logging.getLogger("aiortc").setLevel(level=logging.WARN)
    logging.getLogger("websockets.server").setLevel(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(level=logging.WARNING)
    logging.getLogger("httpcore").setLevel(level=logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(level=logging.WARNING)

    if args.cert_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    wt_ssl_context = None
    if args.wt_cert_file:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        from cryptography.x509 import load_pem_x509_certificate

        with open(args.wt_cert_file, "rb") as f:
            wt_certificate = load_pem_x509_certificate(f.read())
        with open(args.wt_key_file, "rb") as f:
            wt_private_key = load_pem_private_key(f.read(), password=None)
        wt_ssl_context = (wt_certificate, wt_private_key)

    ws_ssl_context = None
    if args.ws_cert_file:
        ws_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ws_ssl_context.load_cert_chain(args.ws_cert_file, args.ws_key_file)

    # Load all scripts from scripts folder
    from script_manager import load_all_scripts

    load_all_scripts()

    asyncio.run(
        run_servers(
            args.host,
            args.port,
            ssl_context,
            args.sip_host,
            args.sip_port,
            args.sip_callback_url,
            args.wt_host,
            args.wt_port,
            wt_ssl_context,
            args.ws_host,
            args.ws_port,
            ws_ssl_context,
            args.rtmp_host,
            args.rtmp_port,
            args.rtmp_callback_url,
            setup=args.setup,
        )
    )
