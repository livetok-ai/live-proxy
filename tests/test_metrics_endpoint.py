import unittest
from unittest.mock import patch, MagicMock
import asyncio
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
import proxy
import metrics


class TestMetricsEndpoint(AioHTTPTestCase):
    """Test cases for the /metrics endpoint."""

    async def get_application(self):
        """Create test application."""
        app = web.Application()
        app.router.add_get("/metrics", proxy.metrics_endpoint)
        return app

    async def test_metrics_endpoint(self):
        """Test the /metrics endpoint returns correct format."""
        # Set some test metrics
        metrics.set_open_connections(2)
        metrics.increment_connection()
        metrics.add_connection_duration(15.5)

        resp = await self.client.request("GET", "/metrics")
        self.assertEqual(resp.status, 200)

        content_type = resp.headers.get("content-type")
        self.assertEqual(content_type, "text/plain; version=0.0.4; charset=utf-8")

        text = await resp.text()

        # Check that metrics are present
        self.assertIn("live_proxy_open_connections", text)
        self.assertIn("live_proxy_total_connections", text)
        self.assertIn("live_proxy_connection_duration_seconds_total", text)

        # Check specific values
        self.assertIn("live_proxy_open_connections 2.0", text)

    async def test_metrics_endpoint_empty(self):
        """Test the /metrics endpoint with no connections."""
        # Reset metrics
        metrics.open_connections_gauge.set(0)

        resp = await self.client.request("GET", "/metrics")
        self.assertEqual(resp.status, 200)

        text = await resp.text()
        self.assertIn("live_proxy_open_connections 0.0", text)


if __name__ == "__main__":
    unittest.main()
