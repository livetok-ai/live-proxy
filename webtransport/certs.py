"""Ephemeral self-signed certificate for local WebTransport testing.

Chrome only accepts a self-signed cert for WebTransport (via
`serverCertificateHashes`) if it is ECDSA and valid for at most 14 days, so
we generate a short-lived EC cert here rather than reusing an arbitrary
long-lived one.
"""

import base64
import datetime
import hashlib

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def generate_self_signed_cert(hostname: str = "localhost", days_valid: int = 13):
    key = ec.generate_private_key(ec.SECP256R1())

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .sign(key, hashes.SHA256())
    )

    return cert, key


def certificate_hash_base64(cert: x509.Certificate) -> str:
    """SHA-256 fingerprint of the DER-encoded cert, as used by serverCertificateHashes."""
    der = cert.public_bytes(serialization.Encoding.DER)
    return base64.b64encode(hashlib.sha256(der).digest()).decode("ascii")
