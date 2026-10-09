import socket
import struct

import msgpack
import pytest
from hypothesis import given
from hypothesis import strategies as st

from contextweave.core.types import JSON
from contextweave.daemon.wire import (
    HEADER,
    MAX_FRAME_BYTES,
    FrameBuffer,
    FrameError,
    Request,
    Response,
    WireError,
    pack_frame,
    parse_request,
    parse_response,
    recv_frame,
    request_message,
    response_message,
    unpack_payload,
)

scalars = (
    st.none()
    | st.booleans()
    | st.integers(-(2**62), 2**62)
    | st.floats(allow_nan=False)
    | st.text()
)
values: st.SearchStrategy[JSON] = st.recursive(
    scalars,
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4)
    ),
    max_leaves=12,
)
messages = st.dictionaries(st.text(max_size=8), values, max_size=5)


@given(messages)
def test_round_trip(message: dict[str, JSON]) -> None:
    assert FrameBuffer().feed(pack_frame(message)) == [message]


@given(st.lists(messages, min_size=1, max_size=4), st.integers(1, 17))
def test_chunked_stream_yields_every_message_in_order(
    batch: list[dict[str, JSON]], step: int
) -> None:
    stream = b"".join(pack_frame(message) for message in batch)
    buffer = FrameBuffer()
    received: list[dict[str, JSON]] = []
    for start in range(0, len(stream), step):
        received.extend(buffer.feed(stream[start : start + step]))
    assert received == batch


def test_header_is_four_byte_big_endian_length() -> None:
    frame = pack_frame({"id": 1})
    (length,) = struct.unpack(">I", frame[:4])
    assert length == len(frame) - 4 == len(msgpack.packb({"id": 1}))


def test_oversized_and_non_map_frames_rejected() -> None:
    with pytest.raises(FrameError):
        FrameBuffer().feed(HEADER.pack(MAX_FRAME_BYTES + 1))
    with pytest.raises(FrameError):
        unpack_payload(msgpack.packb([1, 2]))
    with pytest.raises(FrameError):
        unpack_payload(b"\xc1")
    with pytest.raises(FrameError):
        unpack_payload(msgpack.packb({1: "x"}, use_bin_type=True))


def test_request_and_response_shapes() -> None:
    request = Request(7, "ping", {"a": 1})
    assert parse_request(request_message(request)) == request
    assert parse_request({"id": "x", "method": "ping"}) is None
    ok = Response(7, result={"pid": 1})
    assert parse_response(response_message(ok)) == ok
    failed = Response(7, error=WireError("boom", "bad"))
    assert response_message(failed) == {"id": 7, "error": {"code": "boom", "message": "bad"}}
    assert parse_response(response_message(failed)) == failed


def test_recv_frame_over_socketpair_and_clean_close() -> None:
    left, right = socket.socketpair()
    with left, right:
        left.sendall(pack_frame({"id": 3, "result": "ok"}))
        assert recv_frame(right) == {"id": 3, "result": "ok"}
        left.sendall(pack_frame({"id": 4})[:3])
        left.close()
        with pytest.raises(FrameError):
            recv_frame(right)
    one, two = socket.socketpair()
    with one, two:
        one.close()
        assert recv_frame(two) is None
