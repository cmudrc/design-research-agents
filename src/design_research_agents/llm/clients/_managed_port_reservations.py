"""Internal helpers for reserving managed local-server TCP ports."""

from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(slots=True)
class _ReservedManagedPort:
    """Reserved local TCP port held until a managed server is ready to spawn."""

    port: int
    reservation_socket: socket.socket | None = None

    def release(self) -> None:
        """Release the held socket reservation when present."""
        if self.reservation_socket is None:
            return
        self.reservation_socket.close()
        self.reservation_socket = None


def _resolve_managed_server_port(*, host: str, requested_port: int) -> int:
    """Return a usable local bind port, preferring the requested port when available."""
    requested_probe = _probe_bindable_tcp_port(host=host, port=requested_port)
    if requested_probe == requested_port:
        return requested_port
    fallback_port = _probe_bindable_tcp_port(host=host, port=0)
    return fallback_port if fallback_port is not None else requested_port


def _reserve_managed_server_port(*, host: str, requested_port: int) -> _ReservedManagedPort:
    """Reserve a usable local bind port until the managed server process starts."""
    requested_reservation = _reserve_bindable_tcp_port(host=host, port=requested_port)
    if requested_reservation is not None:
        return requested_reservation
    fallback_reservation = _reserve_bindable_tcp_port(host=host, port=0)
    if fallback_reservation is not None:
        return fallback_reservation
    return _ReservedManagedPort(port=requested_port)


def _probe_bindable_tcp_port(*, host: str, port: int) -> int | None:
    """Return a bindable port for the given host/port pair, or ``None`` when unavailable."""
    reservation = _reserve_bindable_tcp_port(host=host, port=port)
    if reservation is None:
        return None
    resolved_port = reservation.port
    reservation.release()
    return resolved_port


def _reserve_bindable_tcp_port(*, host: str, port: int) -> _ReservedManagedPort | None:
    """Return a held bindable port reservation, or ``None`` when unavailable."""
    try:
        address_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror:
        return None

    for family, socktype, proto, _, sockaddr in address_infos:
        try:
            probe_socket = socket.socket(family, socktype, proto)
            probe_socket.bind(sockaddr)
            bound_port = probe_socket.getsockname()[1]
            if isinstance(bound_port, int):
                return _ReservedManagedPort(port=bound_port, reservation_socket=probe_socket)
            probe_socket.close()
        except OSError:
            continue
    return None


__all__ = [
    "_ReservedManagedPort",
    "_probe_bindable_tcp_port",
    "_reserve_bindable_tcp_port",
    "_reserve_managed_server_port",
    "_resolve_managed_server_port",
]
