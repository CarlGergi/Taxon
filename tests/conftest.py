import ipaddress
import socket

import pytest

from taxon.taxonomy import SEPARATOR, Taxonomy

OFFLINE = (
    "tests run offline; mark a test that needs the network or real data with @pytest.mark.data"
)


def _local(host) -> bool:
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Only tests marked `data` may reach beyond this machine; any other connection fails."""
    if request.node.get_closest_marker("data"):
        return
    connect, lookup = socket.socket.connect, socket.getaddrinfo

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else None
        if sock.family == getattr(socket, "AF_UNIX", None) or _local(host):
            return connect(sock, address)
        raise OSError(f"{OFFLINE} (tried {address!r})")

    def guarded_lookup(host, *args, **kwargs):
        if _local(host):
            return lookup(host, *args, **kwargs)
        raise OSError(f"{OFFLINE} (tried to look up {host!r})")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_lookup)


COLOR = "gid://shopify/TaxonomyAttribute/1"
MATERIAL = "gid://shopify/TaxonomyAttribute/2"


def category(cid, full_name, parent=None, children=(), attributes=()):
    return {
        "id": cid,
        "level": full_name.count(SEPARATOR),
        "name": full_name.split(SEPARATOR)[-1],
        "full_name": full_name,
        "parent_id": parent,
        "attributes": [
            {"id": a, "name": h, "handle": h, "description": "", "extended": h != "color"}
            for h, a in attributes
        ],
        "children": [{"id": c, "name": c} for c in children],
        "ancestors": [],
    }


def toy_release():
    """A small tree in the release format, with the awkward cases on purpose:
    "Bags" under two parents, and "Bags" and "Bag Straps" sharing a string prefix."""
    return {
        "version": "toy-1",
        "verticals": [
            {
                "name": "Apparel",
                "prefix": "ap",
                "categories": [
                    category("ap", "Apparel", children=["ap-1", "ap-2"]),
                    category(
                        "ap-1",
                        "Apparel > Bags",
                        parent="ap",
                        children=["ap-1-1"],
                        attributes=[("color", COLOR), ("strap-color", COLOR)],
                    ),
                    category(
                        "ap-1-1",
                        "Apparel > Bags > Totes",
                        parent="ap-1",
                        attributes=[("color", COLOR), ("material", MATERIAL)],
                    ),
                    category("ap-2", "Apparel > Bag Straps", parent="ap"),
                ],
            },
            {
                "name": "Home",
                "prefix": "ho",
                "categories": [
                    category("ho", "Home", children=["ho-1", "ho-2"]),
                    category("ho-1", "Home > Bags", parent="ho"),
                    category("ho-2", "Home > Kitchen", parent="ho", children=["ho-2-1"]),
                    category("ho-2-1", "Home > Kitchen > Knives", parent="ho-2"),
                ],
            },
        ],
        "attributes": [
            {
                "id": COLOR,
                "name": "Color",
                "handle": "color",
                "description": "",
                "extended_attributes": [{"name": "Strap color", "handle": "strap-color"}],
                "values": [
                    {"id": "v1", "name": "Red", "handle": "red"},
                    {"id": "v2", "name": "Blue", "handle": "blue"},
                ],
            },
            {
                "id": MATERIAL,
                "name": "Material",
                "handle": "material",
                "description": "",
                "extended_attributes": [],
                "values": [{"id": "v3", "name": "Cotton", "handle": "cotton"}],
            },
        ],
    }


@pytest.fixture
def release():
    return toy_release()


@pytest.fixture
def taxonomy():
    return Taxonomy.from_release(toy_release())
