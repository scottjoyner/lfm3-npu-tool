"""
TLS/HTTPS support for LFM3-1.7B inference servers.

Provides:
  * Self-signed certificate generation for development / local deployments
  * OpenSSL config template for production certificate requests
  * SSL context factory for uvicorn-based servers
  * Helper to build HTTPS client URLs

Certificate layout (under ``LFM3_TLS_DIR``, default
``/media/scott/data/lfm3-npu-tool/etc/tls``)::

    etc/tls/
    ├── ca.crt              # CA certificate (self-signed for dev)
    ├── ca.key              # CA private key
    ├── server.crt          # server certificate (signed by CA)
    ├── server.key          # server private key
    └── openssl.cnf         # OpenSSL config

Production deployments should replace ``server.crt``/``server.key`` with a
certificate from a trusted CA and set ``LFM3_TLS_VERIFY_CLIENT=1`` to require
mutual TLS (mTLS) from clients.
"""

import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import ssl

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment)
# ---------------------------------------------------------------------------

TLS_DIR = Path(
    os.environ.get("LFM3_TLS_DIR", "/media/scott/data/lfm3-npu-tool/etc/tls")
)
CA_CERT = TLS_DIR / "ca.crt"
CA_KEY = TLS_DIR / "ca.key"
SERVER_CERT = TLS_DIR / "server.crt"
SERVER_KEY = TLS_DIR / "server.key"
OPENSSL_CNF = TLS_DIR / "openssl.cnf"

TLS_ENABLED = os.environ.get("LFM3_TLS_ENABLED", "false").lower() in ("1", "true", "yes")
TLS_CERT_FILE = os.environ.get("LFM3_TLS_CERT_FILE", str(SERVER_CERT))
TLS_KEY_FILE = os.environ.get("LFM3_TLS_KEY_FILE", str(SERVER_KEY))
TLS_CA_FILE = os.environ.get("LFM3_TLS_CA_FILE", str(CA_CERT))
TLS_VERIFY_CLIENT = os.environ.get("LFM3_TLS_VERIFY_CLIENT", "0").lower() in ("1", "true", "yes")
TLS_PORT_OFFSET = int(os.environ.get("LFM3_TLS_PORT_OFFSET", "0"))  # HTTPS port = HTTP port + offset


def tls_ready() -> bool:
    """Return True if TLS is enabled and cert/key files exist."""
    if not TLS_ENABLED:
        return False
    return Path(TLS_CERT_FILE).exists() and Path(TLS_KEY_FILE).exists()


def ssl_context(client_auth: bool = False) -> Optional[ssl.SSLContext]:
    """
    Build an SSLContext for uvicorn.

    Returns None when TLS is disabled (caller should run plain HTTP).
    When ``client_auth`` is True (mTLS), the server requires and verifies
    client certificates against the configured CA.
    """
    if not tls_ready():
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(TLS_CERT_FILE, TLS_KEY_FILE)

    if client_auth and TLS_VERIFY_CLIENT:
        ctx.load_verify_locations(TLS_CA_FILE)
        ctx.verify_mode = ssl.CERT_REQUIRED
    else:
        ctx.verify_mode = ssl.CERT_NONE

    # Modern, server-hardened cipher suite
    ctx.set_ciphers(
        "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:"
        "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:"
        "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256"
    )
    # Disable legacy TLS (TLS 1.0/1.1) using modern API
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def https_url(host: str, port: int, path: str = "") -> str:
    """Build an HTTPS (or HTTP when TLS disabled) URL."""
    scheme = "https" if TLS_ENABLED else "http"
    effective_port = port + TLS_PORT_OFFSET if TLS_ENABLED else port
    return f"{scheme}://{host}:{effective_port}{path}"


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

OPENSSL_CNF_TEMPLATE = """[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_ext
prompt = no

[req_distinguished_name]
C = US
ST = California
L = San Francisco
O = LFM3-1.7B
OU = Inference
CN = localhost

[v3_ext]
subjectAltName = @alt_names
basicConstraints = critical, CA:TRUE
keyUsage = digitalSignature, keyEncipherment, keyCertSign, cRLSign
extendedKeyUsage = serverAuth, clientAuth

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
DNS.3 = xwing
IP.1 = 127.0.0.1
"""


def generate_dev_certificates(force: bool = False) -> Tuple[bool, str]:
    """
    Generate a self-signed CA + server certificate for local development.

    Returns (success, message).
    """
    TLS_DIR.mkdir(parents=True, exist_ok=True)

    # Write OpenSSL config
    cnf_path = OPENSSL_CNF
    if not cnf_path.exists() or force:
        cnf_path.write_text(OPENSSL_CNF_TEMPLATE)
        cnf_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # CA key + cert (self-signed root CA, 10 years)
    if force or not CA_KEY.exists() or not CA_CERT.exists():
        _run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", str(CA_KEY), "-out", str(CA_CERT),
                "-days", "3650", "-nodes", "-subj",
                "/C=US/ST=California/L=San Francisco/O=LFM3-1.7B/OU=CA/CN=LFM3 Root CA",
                "-config", str(cnf_path),
                "-extensions", "v3_ext",
            ]
        )
        CA_KEY.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # Server key + CSR (2048-bit, SANs for localhost/127.0.0.1)
    csr_path = Path(tempfile.mktemp(suffix=".csr"))
    if force or not SERVER_KEY.exists():
        _run(
            [
                "openssl", "req", "-newkey", "rsa:2048",
                "-keyout", str(SERVER_KEY), "-out", str(csr_path),
                "-nodes", "-subj",
                "/C=US/ST=California/L=San Francisco/O=LFM3-1.7B/OU=Inference/CN=localhost",
                "-config", str(cnf_path),
            ]
        )
        SERVER_KEY.chmod(stat.S_IRUSR | stat.S_IWUSR)
    else:
        # Reuse existing key, generate a fresh CSR from it
        _run(
            [
                "openssl", "req", "-new",
                "-key", str(SERVER_KEY), "-out", str(csr_path),
                "-subj",
                "/C=US/ST=California/L=San Francisco/O=LFM3-1.7B/OU=Inference/CN=localhost",
                "-config", str(cnf_path),
            ]
        )

    # Sign server cert with CA (2 years, SANs)
    ext_path = Path(tempfile.mktemp(suffix=".ext"))
    ext_path.write_text(
        "subjectAltName=DNS:localhost,DNS:127.0.0.1,DNS:xwing,IP:127.0.0.1\n"
        "extendedKeyUsage=serverAuth\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
    )
    if force or not SERVER_CERT.exists():
        _run(
            [
                "openssl", "x509", "-req",
                "-in", str(csr_path),
                "-CA", str(CA_CERT), "-CAkey", str(CA_KEY),
                "-CAcreateserial",
                "-out", str(SERVER_CERT),
                "-days", "825",
                "-sha256",
                "-extfile", str(ext_path),
            ]
        )

    # Cleanup temp artifacts
    for p in (csr_path, ext_path):
        try:
            p.unlink()
        except OSError:
            pass

    return True, f"Certificates generated in {TLS_DIR}"


def _run(cmd: list) -> None:
    """Run a command, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def tls_status() -> dict:
    """Return a dict describing TLS state (for /health /status endpoints)."""
    return {
        "tls_enabled": TLS_ENABLED,
        "tls_ready": tls_ready(),
        "cert_file": TLS_CERT_FILE,
        "key_file": TLS_KEY_FILE,
        "ca_file": TLS_CA_FILE,
        "verify_client": TLS_VERIFY_CLIENT,
        "cert_exists": Path(TLS_CERT_FILE).exists(),
        "key_exists": Path(TLS_KEY_FILE).exists(),
        "ca_exists": CA_CERT.exists(),
    }
