import asyncio
import logging
import os
import re
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

from connection import Connection, ConnectionManager

logger = logging.getLogger(__name__)

connections = ConnectionManager()


@dataclass
class SIPSession:
    """Data structure to store SIP session information."""

    uri: str
    session_id: str
    connection: Optional[Connection] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_message_at: datetime = field(default_factory=datetime.now)


class SIPServer:
    """TCP server that handles SIP-like message parsing."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5060, callback_url: Optional[str] = None):
        """
        Initialize the SIP server.

        Args:
            host: Host address to bind to
            port: Port number to listen on (default: 5060, standard SIP port)
            callback_url: URL to call when receiving INVITE messages
        """
        self.host = host
        self.port = port
        self.callback_url = callback_url
        self.server: Optional[asyncio.Server] = None
        self.sessions: Dict[str, SIPSession] = {}

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handle incoming client connection.

        Args:
            reader: Stream reader for receiving data
            writer: Stream writer for sending data
        """
        addr = writer.get_extra_info("peername")
        logger.info(f"New connection from {addr}")

        try:
            while True:
                # Read data until we find the message terminator \r\n\r\n
                message = await self.read_message(reader)

                if not message:
                    logger.info(f"Connection closed by {addr}")
                    break

                # Parse the message into header lines and body
                lines, body = self.parse_message(message)

                logger.info(f"Received message from {addr} with {len(lines)} header lines")
                for i, line in enumerate(lines):
                    logger.debug(f"Line {i}: {line}")
                if body:
                    logger.debug(f"Body length: {len(body)} bytes")

                # Process the message
                await self.process_message(lines, body, writer)

        except asyncio.CancelledError:
            logger.info(f"Connection with {addr} cancelled")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Connection with {addr} closed")

    async def read_message(self, reader: asyncio.StreamReader) -> str:
        """
        Read a complete message from the stream, including body if Content-Length is present.
        Headers are terminated by \\r\\n\\r\\n, followed by body.

        Args:
            reader: Stream reader to read from

        Returns:
            Complete message as a string, or empty string if connection closed
        """
        buffer = b""
        terminator = b"\r\n\r\n"

        # First read the headers
        while True:
            try:
                chunk = await reader.read(1024)
                if not chunk:
                    # Connection closed
                    return ""

                buffer += chunk

                # Check if we have the complete headers
                if terminator in buffer:
                    # Find the position of the terminator
                    pos = buffer.find(terminator)
                    headers_part = buffer[: pos + 4]  # Include \r\n\r\n
                    body_part = buffer[pos + 4 :]  # Everything after headers

                    # Parse headers to find Content-Length
                    headers_str = headers_part.decode("utf-8")
                    content_length = 0

                    for line in headers_str.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            try:
                                content_length = int(line.split(":", 1)[1].strip())
                            except ValueError:
                                pass
                            break

                    # Read the body if Content-Length is present
                    if content_length > 0:
                        while len(body_part) < content_length:
                            chunk = await reader.read(content_length - len(body_part))
                            if not chunk:
                                break
                            body_part += chunk

                        # Return headers + body
                        return (headers_part + body_part).decode("utf-8")
                    else:
                        # No body, return just headers (excluding final \r\n\r\n)
                        return buffer[: pos + 2].decode("utf-8")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error reading message: {e}")
                raise

    def parse_message(self, message: str) -> tuple[List[str], Optional[str]]:
        """
        Parse a message into individual header lines and body.
        Each line is terminated by \\r\\n.

        Args:
            message: Raw message string

        Returns:
            Tuple of (header_lines, body) where body is None if no body present
        """
        # Check if there's a body (headers and body separated by \r\n\r\n)
        if "\r\n\r\n" in message:
            headers_part, body = message.split("\r\n\r\n", 1)
            lines = [line for line in headers_part.split("\r\n") if line]
            return lines, body if body else None
        else:
            # No body, just headers
            lines = [line for line in message.split("\r\n") if line]
            return lines, None

    def extract_sip_info(self, lines: List[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract SIP method, URI, and session ID from message lines.

        Args:
            lines: Parsed message lines

        Returns:
            Tuple of (method, uri, session_id) or (None, None, None) if parsing fails
        """
        if not lines:
            return None, None, None

        # First line contains: METHOD URI SIP/2.0
        # Example: INVITE sip:testuser@localhost:5060;transport=tcp SIP/2.0
        first_line = lines[0].split()
        if len(first_line) < 2:
            return None, None, None

        method = first_line[0]
        uri = first_line[1]

        # Find Call-ID header (this is the session ID)
        session_id = None
        for line in lines[1:]:
            if line.startswith("Call-ID:") or line.startswith("i:"):
                session_id = line.split(":", 1)[1].strip()
                break

        return method, uri, session_id

    async def send_sip_response(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        reason: str,
        call_id: str,
        via: str,
        from_hdr: str,
        to_hdr: str,
        cseq: str,
        body: Optional[str] = None,
    ):
        """
        Send a SIP response message.

        Args:
            writer: Stream writer for sending data
            status_code: HTTP-like status code (e.g., 200, 404)
            reason: Reason phrase (e.g., "OK", "Not Found")
            call_id: Call-ID from the request
            via: Via header from the request
            from_hdr: From header from the request
            to_hdr: To header from the request
            cseq: CSeq header from the request
            body: Optional body content (e.g., SDP)
        """
        # Get the server socket address for Contact header
        contact_host = self.host if self.host != "0.0.0.0" else "localhost"
        contact_port = self.port

        contact_uri = f"<sip:{contact_host}:{contact_port};transport=tcp>"

        response = f"SIP/2.0 {status_code} {reason}\r\n"
        response += f"Via: {via}\r\n"
        response += f"From: {from_hdr}\r\n"
        response += f"To: {to_hdr}\r\n"
        response += f"Call-ID: {call_id}\r\n"
        response += f"CSeq: {cseq}\r\n"
        response += f"Contact: {contact_uri}\r\n"

        if body:
            response += f"Content-Type: application/sdp\r\n"
            response += f"Content-Length: {len(body)}\r\n"
            response += "\r\n"
            response += body
        else:
            response += f"Content-Length: 0\r\n"
            response += "\r\n"

        writer.write(response.encode("utf-8"))
        await writer.drain()
        logger.info(
            f"Sent SIP response: {status_code} {reason}" + (f" with SDP body ({len(body)} bytes)" if body else "")
        )

    def add_to_tag(self, to_hdr: str, tag: str) -> str:
        """
        Add a tag parameter to the To header if it doesn't already have one.
        The tag parameter is a header parameter and must go OUTSIDE angle brackets.

        Args:
            to_hdr: Original To header
            tag: Tag value to add

        Returns:
            To header with tag parameter

        Examples:
            "Display Name" <sip:user@host> -> "Display Name" <sip:user@host>;tag=xyz
            <sip:user@host> -> <sip:user@host>;tag=xyz
            sip:user@host -> sip:user@host;tag=xyz
        """
        # Check if the To header already has a tag
        if ";tag=" in to_hdr:
            return to_hdr

        # The tag parameter is a header parameter, so it always goes after the URI
        # (outside angle brackets if present)
        return f"{to_hdr};tag={tag}"

    def extract_headers(self, lines: List[str]) -> Dict[str, str]:
        """
        Extract common SIP headers from message lines.

        Args:
            lines: Parsed message lines

        Returns:
            Dictionary of header names to values
        """
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()
        return headers

    async def make_callback(self, uri: str, from_uri: str) -> tuple[bool, Optional[Dict]]:
        """
        Make HTTP callback to the configured callback URL.

        Args:
            uri: SIP URI to send in the callback (the To URI)
            from_uri: SIP From URI (caller information)

        Returns:
            Tuple of (success, response_data) where response_data contains connection parameters
        """
        if not self.callback_url:
            logger.warning("No callback URL configured, skipping callback only for testing purposes")
            return True, {}

        try:
            callback_payload = {"uri": uri, "from": from_uri}

            logger.info(f"Making callback to {self.callback_url} with payload: {callback_payload}")

            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Create TCP connector with SSL context
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self.callback_url, json=callback_payload, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    logger.info(f"Callback response: {response.status}")
                    if 200 <= response.status < 300:
                        try:
                            response_data = await response.json()
                            # logger.info(f"Callback response data: {response_data}")
                            return True, response_data
                        except Exception as e:
                            logger.error(f"Error parsing callback response: {e}")
                            return True, {}
                    return False, None
        except Exception as e:
            logger.error(f"Error making callback to {self.callback_url}: {e}")
            return False, None

    async def process_message(self, lines: List[str], body: Optional[str], writer: asyncio.StreamWriter):
        """
        Process a parsed message and send response.

        Args:
            lines: Parsed message header lines
            body: Message body (e.g., SDP for INVITE)
            writer: Stream writer for sending responses
        """
        if not lines:
            return

        # Extract SIP info
        method, uri, session_id = self.extract_sip_info(lines)

        if not method or not uri:
            logger.warning("Could not extract method or URI from message")
            return

        logger.info(f"Processing {method} message - URI: {uri}, Session ID: {session_id}")

        # Extract headers for response
        headers = self.extract_headers(lines)
        via = headers.get("Via", "")
        from_hdr = headers.get("From", "")
        to_hdr = headers.get("To", "")
        cseq = headers.get("CSeq", "")

        # Extract From URI (extract just the URI part from From header)
        from_uri = None
        if from_hdr:
            # From header format: "Display Name" <sip:user@host> or <sip:user@host>
            match = re.search(r"<([^>]+)>", from_hdr)
            if match:
                from_uri = match.group(1)
            else:
                # If no angle brackets, use the whole header
                from_uri = from_hdr.split(";")[0].strip()

        # Handle INVITE
        if method == "INVITE":
            if session_id and session_id in self.sessions:
                logger.info(f"Session {session_id} already exists, ignoring duplicate INVITE")
                return

            if not body:
                logger.warning("No SDP body in INVITE message")
                if session_id and via and from_hdr and to_hdr and cseq:
                    await self.send_sip_response(writer, 400, "Bad Request", session_id, via, from_hdr, to_hdr, cseq)
                return

            # Generate a tag for this dialog (using session_id for uniqueness)
            to_tag = session_id[:8] if session_id else "12345678"
            to_hdr_with_tag = self.add_to_tag(to_hdr, to_tag)

            # Send 180 Ringing immediately before callback
            if session_id and via and from_hdr and to_hdr and cseq:
                logger.info("Sending 180 Ringing response immediately for INVITE")
                await self.send_sip_response(writer, 180, "Ringing", session_id, via, from_hdr, to_hdr_with_tag, cseq)

            # Make HTTP callback
            logger.info(f"Making callback for INVITE to {uri} from {from_uri}")
            callback_success, callback_data = await self.make_callback(uri, from_uri)

            if callback_success and callback_data:
                # Extract connection parameters from callback response
                model = callback_data.get("model", "gemini")
                system_instructions = callback_data.get("system_instructions")
                tools = callback_data.get("tools")
                voice = callback_data.get("voice")
                language = callback_data.get("language")
                api_key = callback_data.get("api_key")

                logger.info(f"Starting connection with model={model}")

                conn_info = connections.create_connection(
                    callback=self.callback_url, metadata=callback_data, public_ip=self.host
                )
                connection = conn_info.connection

                try:
                    # Start connection and get SDP response
                    sdp_response = await connection.start(
                        sdp=body,
                        model=model,
                        system_instructions=system_instructions,
                        tools=tools,
                        voice=voice,
                        language=language,
                        api_key=api_key,
                    )

                    # Create new session with connection
                    if session_id:
                        session = SIPSession(uri=uri, session_id=session_id, connection=connection)
                        self.sessions[session_id] = session

                        logger.info(f"Created new session {session_id} for URI {uri} with connection {connection.id}")
                        logger.info(f"Active sessions: {len(self.sessions)}")

                    # Send 200 OK response with SDP (use tagged To header)
                    if session_id and via and from_hdr and to_hdr and cseq and sdp_response:
                        await self.send_sip_response(
                            writer, 200, "OK", session_id, via, from_hdr, to_hdr_with_tag, cseq, body=sdp_response
                        )
                    else:
                        logger.warning("Missing headers or SDP for sending 200 OK response")
                        await self.send_sip_response(
                            writer, 500, "Internal Server Error", session_id, via, from_hdr, to_hdr_with_tag, cseq
                        )

                except Exception as e:
                    logger.error(f"Error starting connection: {e}", exc_info=True)
                    if session_id and via and from_hdr and to_hdr and cseq:
                        await self.send_sip_response(
                            writer, 500, "Internal Server Error", session_id, via, from_hdr, to_hdr_with_tag, cseq
                        )
            else:
                # Send 503 Service Unavailable if callback failed
                if session_id and via and from_hdr and to_hdr and cseq:
                    await self.send_sip_response(
                        writer, 503, "Service Unavailable", session_id, via, from_hdr, to_hdr_with_tag, cseq
                    )
                    logger.warning("Callback failed, sent 503 response")

        # Handle BYE
        elif method == "BYE":
            if session_id and session_id in self.sessions:
                # Remove session and close connection
                session = self.sessions.pop(session_id)
                logger.info(f"Removed session {session_id} for URI {session.uri}")

                # Close connection if exists
                if session.connection:
                    try:
                        await session.connection.close()
                        logger.info(f"Closed connection {session.connection.id}")
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")

                logger.info(f"Active sessions: {len(self.sessions)}")

                # Send 200 OK response
                if via and from_hdr and to_hdr and cseq:
                    await self.send_sip_response(writer, 200, "OK", session_id, via, from_hdr, to_hdr, cseq)
            else:
                logger.warning(f"Session {session_id} not found for BYE message")

        # Update session timestamp if it exists
        if session_id and session_id in self.sessions:
            self.sessions[session_id].last_message_at = datetime.now()

    async def start(self):
        """Start the TCP server."""
        self.server = await asyncio.start_server(self.handle_client, "0.0.0.0", self.port)

        addr = self.server.sockets[0].getsockname()
        logger.info(f"SIP server listening on {addr[0]}:{addr[1]}")
        if self.callback_url:
            logger.info(f"Callback URL: {self.callback_url}")

        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the TCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("SIP server stopped")


async def main():
    """Main entry point for running the SIP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure server with custom port if needed
    port = int(os.getenv("SIP_PORT", "5060"))
    host = os.getenv("SIP_HOST", "0.0.0.0")
    callback_url = os.getenv("SIP_CALLBACK_URL")

    server = SIPServer(host=host, port=port, callback_url=callback_url)

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
