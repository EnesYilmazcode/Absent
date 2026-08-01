"""Self-signed cert so the phone's browser will hand over its camera.

Safari only exposes getUserMedia on a secure page, and there is no internet at
demo time to get a real certificate. Covers every local address plus Apple's
USB/hotspot range, since the tether address is not known until the cable is in.
"""

import datetime
import ipaddress
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def local_ips():
    ips = {"127.0.0.1"}
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ips.add(info[4][0])
    # Apple hands out 172.20.10.x over USB tethering and Personal Hotspot.
    ips.update(f"172.20.10.{n}" for n in range(1, 15))
    return sorted(ips)


def main():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "absent.local")])
    alts = [x509.DNSName("localhost"), x509.DNSName("absent.local")]
    alts += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in local_ips()]

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alts), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    print("wrote cert.pem and key.pem for:")
    for ip in local_ips():
        print("  ", ip)


if __name__ == "__main__":
    main()
