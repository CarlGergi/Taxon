"""The offline tests cannot reach the network, so CI needs no network, keys or downloads."""

import socket

import pytest


def test_a_connection_beyond_this_machine_fails():
    with socket.socket() as sock, pytest.raises(OSError, match="tests run offline"):
        sock.connect(("192.0.2.1", 443))
    with pytest.raises(OSError, match="tests run offline"):
        socket.getaddrinfo("example.com", 443)


def test_this_machine_is_still_reachable():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5):
            pass
    assert socket.getaddrinfo("localhost", 80)
