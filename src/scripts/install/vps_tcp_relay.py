import socket
import sys
import threading


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
    listen_port = int(sys.argv[1])
    target_port = int(sys.argv[2])
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", listen_port))
    server.listen()
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client, target_port), daemon=True).start()


if __name__ == "__main__":
    main()