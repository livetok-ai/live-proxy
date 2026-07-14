import pytest
from aiohttp.test_utils import AioHTTPTestCase

import proxy
from connection import Connection, ConnectionManager, SendingTrack
from session import SessionEvents, SessionManager


class FakeChannel:
    def __init__(self, label="data"):
        self.label = label
        self.readyState = "open"
        self.messages = []

    def send(self, message):
        self.messages.append(message)


@pytest.fixture(autouse=True)
def clean_managers():
    SessionManager().sessions.clear()
    ConnectionManager().connections.clear()
    yield
    SessionManager().sessions.clear()
    ConnectionManager().connections.clear()


def test_create_connection_attaches_to_new_session():
    conn_info = ConnectionManager().create_connection()
    connection = conn_info.connection

    assert connection.session is not None
    assert connection in connection.session.connections
    assert SessionManager().find_session_by_id(connection.session.id) is connection.session


def test_create_connection_attaches_to_existing_session():
    manager = ConnectionManager()
    first = manager.create_connection().connection
    second = manager.create_connection(session=first.session).connection

    assert second.session is first.session
    assert len(first.session.connections) == 2
    assert len(SessionManager().list_sessions()) == 1


async def test_session_destroyed_when_last_connection_closes():
    manager = ConnectionManager()
    first_info = manager.create_connection()
    session = first_info.connection.session
    second_info = manager.create_connection(session=session)

    await first_info.connection.close()
    assert SessionManager().find_session_by_id(session.id) is session

    await second_info.connection.close()
    assert SessionManager().find_session_by_id(session.id) is None
    assert session.connections == []


def test_session_destroyed_when_connection_removed_before_start():
    manager = ConnectionManager()
    conn_info = manager.create_connection()
    session = conn_info.connection.session

    manager.remove_connection(conn_info)

    assert SessionManager().find_session_by_id(session.id) is None


def test_datachannel_message_broadcast_to_other_connections():
    session = SessionManager().create_session()
    sender = Connection()
    receiver = Connection()
    session.attach(sender)
    session.attach(receiver)

    sender_channel = FakeChannel()
    receiver_channel = FakeChannel()
    sender.data_channels.append(sender_channel)
    receiver.data_channels.append(receiver_channel)

    sender._broadcast_to_channels('{"type": "transcription"}')

    assert receiver_channel.messages == ['{"type": "transcription"}']
    # The sender only gets it through its own channels, not back from the session
    assert sender_channel.messages == ['{"type": "transcription"}']


def test_video_event_delivered_to_subscribed_connections():
    session = SessionManager().create_session()
    source = Connection()
    viewer = Connection()
    session.attach(source)
    session.attach(viewer)
    viewer.send_video_track = SendingTrack("video")

    frame = object()
    source._publish_to_session(SessionEvents.VIDEO, frame)

    assert viewer.send_video_track.queue.get_nowait() is frame


def test_audio_event_delivered_to_subscribed_connections():
    session = SessionManager().create_session()
    source = Connection()
    viewer = Connection()
    session.attach(source)
    session.attach(viewer)

    frame = object()
    source._publish_to_session(SessionEvents.AUDIO, frame)

    assert viewer.output_audio_queue.get_nowait() is frame
    assert source.output_audio_queue.empty()


class TestSessionEndpoints(AioHTTPTestCase):
    async def get_application(self):
        server = proxy.HTTPServer(host="localhost", port=8080, ssl_context=None)
        return server.app

    async def setUpAsync(self):
        await super().setUpAsync()
        SessionManager().sessions.clear()
        ConnectionManager().connections.clear()

    async def test_list_sessions_empty(self):
        resp = await self.client.request("GET", "/session")
        assert resp.status == 200
        data = await resp.json()
        assert data["sessions"] == []

    async def test_list_sessions(self):
        conn_info = proxy.connections.create_connection()
        session = conn_info.connection.session

        resp = await self.client.request("GET", "/session")
        assert resp.status == 200
        data = await resp.json()

        entry = next(s for s in data["sessions"] if s["id"] == session.id)
        assert entry["connections"] == [conn_info.connection.id]
        assert entry["created_at"] > 0

        proxy.connections.remove_connection(conn_info)

    async def test_join_unknown_session_returns_404(self):
        resp = await self.client.request("POST", "/session/unknown/connection", json={"sdp": "v=0"})
        assert resp.status == 404

    async def test_join_session_requires_sdp(self):
        conn_info = proxy.connections.create_connection()
        session_id = conn_info.connection.session.id

        resp = await self.client.request("POST", f"/session/{session_id}/connection", json={})
        assert resp.status == 400

        proxy.connections.remove_connection(conn_info)
