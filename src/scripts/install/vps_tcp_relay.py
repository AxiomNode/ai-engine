"""Minimal TCP relay used by the STG VPS reverse-tunnel topology.

Usage::

    python3 vps_tcp_relay.py <listen_port> <target_port> [--bind HOST]

The relay forwards raw TCP to ``127.0.0.1:<target_port>`` (the loopback end of
the SSH reverse tunnel). It performs NO authentication of its own: anything
that can reach the listening socket can reach ai-engine. For that reason it
binds to loopback (``127.0.0.1``) by default. Binding a non-loopback address
must be requested explicitly with ``--bind`` or ``AI_ENGINE_RELAY_BIND_HOST``
and logs a warning. When you do expose it, restrict the port with a VPS
firewall to the cluster networks and set ``AI_ENGINE_API_KEY`` on ai-engine so
the HTTP layer rejects unauthenticated callers.
"""

import ipaddress
import os
import socket
import sys
import threading

BIND_ENV_VAR = "AI_ENGINE_RELAY_BIND_HOST"
DEFAULT_BIND_HOST = "127.0.0.1"


def is_loopback_host(host: str) -> bool:
    """Return True when *host* is a loopback name or address."""
    value = host.strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def parse_args(argv: list[str], environ: dict[str, str]) -> tuple[int, int, str]:
    """Parse ``argv`` (without program name) into (listen, target, bind_host)."""
    bind_host = environ.get(BIND_ENV_VAR, "").strip() or DEFAULT_BIND_HOST
    positional: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--bind":
            index += 1
            if index >= len(argv) or not argv[index].strip():
                raise SystemExit("--bind requires a host value")
            bind_host = argv[index].strip()
        elif arg.startswith("--bind="):
            bind_host = arg.split("=", 1)[1].strip()
            if not bind_host:
                raise SystemExit("--bind requires a host value")
        else:
            positional.append(arg)
        index += 1
    if len(positional) != 2:
        raise SystemExit(
            "usage: vps_tcp_relay.py <listen_port> <target_port> [--bind HOST]"
        )
    return int(positional[0]), int(positional[1]), bind_host


def pipe(source: socket.socket, destination: socket.socket) -> None:
    try:
        while True:
            payload = source.recv(65536)
            if not payload:
                break
            destination.sendall(payload)
    except Exception:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def handle(client: socket.socket, target_port: int) -> None:
    upstream = socket.create_connection(("127.0.0.1", target_port))
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


def main() -> None:
    listen_port, target_port, bind_host = parse_args(sys.argv[1:], dict(os.environ))
    if not is_loopback_host(bind_host):
        print(
            f"WARNING: relay binding to non-loopback address {bind_host}:"
            f"{listen_port} with NO authentication. Restrict access with a "
            "firewall and set AI_ENGINE_API_KEY on ai-engine.",
            file=sys.stderr,
            flush=True,
        )
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    server = socket.socket(family, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, listen_port))
    server.listen()
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client, target_port), daemon=True).start()


if __name__ == "__main__":
    main()
