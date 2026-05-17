import unittest

import metrics


class TestMetrics(unittest.TestCase):
    """Test cases for the metrics module."""

    def setUp(self):
        # Reset metrics between tests
        metrics.open_connections_gauge.set(0)
        metrics.total_connections_counter._value._value = 0
        metrics.connection_duration_counter._value._value = 0

    def test_set_open_connections(self):
        """Test setting open connections count."""
        metrics.set_open_connections(5)
        self.assertEqual(metrics.open_connections_gauge._value._value, 5)

        metrics.set_open_connections(0)
        self.assertEqual(metrics.open_connections_gauge._value._value, 0)

    def test_increment_connection(self):
        """Test incrementing total connections counter."""
        initial_value = metrics.total_connections_counter._value._value
        metrics.increment_connection()
        self.assertEqual(metrics.total_connections_counter._value._value, initial_value + 1)

        metrics.increment_connection()
        self.assertEqual(metrics.total_connections_counter._value._value, initial_value + 2)

    def test_add_connection_duration(self):
        """Test adding connection duration."""
        initial_value = metrics.connection_duration_counter._value._value
        metrics.add_connection_duration(10.5)
        self.assertEqual(metrics.connection_duration_counter._value._value, initial_value + 10.5)

        metrics.add_connection_duration(5.0)
        self.assertEqual(metrics.connection_duration_counter._value._value, initial_value + 15.5)

    def test_get_metrics(self):
        """Test getting metrics output."""
        metrics.set_open_connections(3)
        metrics.increment_connection()
        metrics.add_connection_duration(12.5)

        output = metrics.get_metrics()
        self.assertIsInstance(output, bytes)
        output_str = output.decode("utf-8")

        # Check that our metrics are present
        self.assertIn("live_proxy_open_connections", output_str)
        self.assertIn("live_proxy_total_connections", output_str)
        self.assertIn("live_proxy_connection_duration_seconds_total", output_str)

        # Check values
        self.assertIn("live_proxy_open_connections 3.0", output_str)

    def test_get_content_type(self):
        """Test getting content type."""
        content_type = metrics.get_content_type()
        self.assertEqual(content_type, metrics.CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    unittest.main()
