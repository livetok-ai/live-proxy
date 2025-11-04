import socket


def get_public_ip() -> str:
    """
    Get the public IP address of this machine.

    Returns:
        The first non-localhost IP address found, or "127.0.0.1" as fallback
    """
    try:
        hostname = socket.gethostname()
        # Get all IP addresses associated with the hostname
        all_ips = socket.gethostbyname_ex(hostname)[2]
        # Filter out localhost and loopback addresses
        non_localhost_ips = [ip for ip in all_ips if not ip.startswith("127.") and not ip.startswith("::")]
        if non_localhost_ips:
            return non_localhost_ips[0]
    except Exception:
        pass
    # Ultimate fallback to localhost
    return "127.0.0.1"
