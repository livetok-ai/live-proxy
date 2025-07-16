"""Prometheus metrics for live-proxy."""

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Metrics
open_connections_gauge = Gauge(
    "live_proxy_open_connections", "Number of currently open WebRTC connections"
)

total_connections_counter = Counter(
    "live_proxy_total_connections", "Total number of WebRTC connections created"
)

connection_duration_counter = Counter(
    "live_proxy_connection_duration_seconds_total",
    "Total duration of all connections in seconds",
)


def get_metrics():
    """Get current metrics in Prometheus format."""
    return generate_latest()


def get_content_type():
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST


def increment_connection():
    """Increment the total connection counter."""
    total_connections_counter.inc()


def set_open_connections(count):
    """Set the current number of open connections."""
    open_connections_gauge.set(count)


def add_connection_duration(duration):
    """Add connection duration to the total."""
    connection_duration_counter.inc(duration)
