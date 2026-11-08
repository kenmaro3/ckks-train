"""Runtime-only framed requests to the actual same-executable supervisor.

This module cannot spawn or signal a process. An inherited socket/hold is custody,
not an authority constructor; the supervisor independently validates every request.
"""
import os
import socket
import stat
import struct

CAP = 8 * 1024 * 1024
_HOLD = None


class Invalid(ValueError):
    """A bounded private protocol violation."""


def require(ok, message):
    if not ok:
        raise Invalid(message)


class Broker:
    def __init__(self, stream, send_sequence, receive_sequence, parent_pid=None):
        self.stream = stream
        self.send_sequence = send_sequence
        self.receive_sequence = receive_sequence
        self.parent_pid = parent_pid
        self.parent_uid = os.geteuid() if parent_pid is not None else None

    def check_parent(self):
        if self.parent_pid is not None:
            require(os.getppid() == self.parent_pid and os.getuid() == os.geteuid() == self.parent_uid, "runtime parent/account changed")

    def send(self, tag, payload):
        self.check_parent()
        require(len(payload) <= CAP and self.send_sequence < 2**64 - 1, "broker send bounds")
        self.stream.sendall(struct.pack("<IBQ", 9 + len(payload), tag, self.send_sequence) + payload)
        self.send_sequence += 1

    def exact(self, count):
        require(0 <= count <= CAP + 9, "broker read bound")
        data = bytearray()
        while len(data) < count:
            part = self.stream.recv(min(65536, count - len(data)))
            require(bool(part), "supervisor EOF; stop at boundary")
            data.extend(part)
        return bytes(data)

    def receive(self):
        self.check_parent()
        length = struct.unpack("<I", self.exact(4))[0]
        require(9 <= length <= CAP + 9, "broker frame length")
        frame = self.exact(length)
        tag, sequence = struct.unpack("<BQ", frame[:9])
        require(tag in (4, 7) and sequence == self.receive_sequence and sequence < 2**64 - 1, "broker reply order")
        self.receive_sequence += 1
        require(tag != 7, "supervisor stop")
        return frame[9:]

    def run(self, argv, cwd, env, cap):
        del env  # The parent uses only its retained approved environment.
        raw = [os.fsencode(arg) for arg in argv]
        require(0 < len(raw) <= 256 and all(len(v) <= 4096 and b"\0" not in v for v in raw), "broker argv bounds")
        directory = os.fsencode(cwd)
        require(len(directory) <= 4096 and directory.startswith(b"/") and b"\0" not in directory, "broker root bounds")
        payload = bytearray(b"\x01")  # Exact Git broker kind; worker alone may request Python.
        payload.extend(struct.pack("<Q", len(directory)) + directory)
        payload.extend(struct.pack("<Q", len(raw)))
        for item in raw:
            payload.extend(struct.pack("<Q", len(item)) + item)
        self.send(3, payload)
        reply = self.receive()
        require(len(reply) >= 8 and struct.unpack("<Q", reply[:8])[0] == 0, "broker Git failed")
        require(len(reply) - 8 <= cap, "broker output cap")
        return reply[8:]

    def result(self, output):
        self.send(4, struct.pack("<Q", 0) + output)


def acquire(device, inode, parent_pid):
    """Check inherited descriptor continuity after the Rust trampoline handshake."""
    global _HOLD
    actual = os.fstat(0)
    require(stat.S_ISREG(actual.st_mode) and (actual.st_dev, actual.st_ino) == (device, inode), "runtime inherited lock")
    require(stat.S_IMODE(actual.st_mode) == 0o600 and actual.st_uid == os.geteuid() and actual.st_nlink == 1, "runtime lock owner/mode")
    require(os.getppid() == parent_pid and os.getuid() == os.geteuid(), "runtime parent/account")
    stream = socket.socket(fileno=os.dup(1))
    require(stream.family == socket.AF_UNIX and stream.type == socket.SOCK_STREAM, "runtime private control socket")
    pid, uid, _gid = struct.unpack("3i", stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    require(pid == parent_pid and uid == os.geteuid(), "runtime supervisor peer")
    require(_HOLD is None, "duplicate runtime acquisition")
    _HOLD = os.dup(0)  # Intentionally held to process exit; never unlock or close here.
    stream.setblocking(True)
    return Broker(stream, 2, 2, parent_pid=parent_pid)
