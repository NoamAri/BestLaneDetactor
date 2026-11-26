import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from lane_intel.serve import RangeHandler, parse_range


@pytest.mark.parametrize("header,size,expected", [
    ("bytes=0-99", 1000, (0, 99)),
    ("bytes=500-", 1000, (500, 999)),
    ("bytes=-100", 1000, (900, 999)),
    ("bytes=900-5000", 1000, (900, 999)),
    ("bytes=1000-", 1000, None),
    ("bytes=-", 1000, None),
    (None, 1000, None),
])
def test_parse_range(header, size, expected):
    assert parse_range(header, size) == expected


def test_server_answers_range_requests(tmp_path):
    data = bytes(range(256)) * 40
    (tmp_path / "video.mp4").write_bytes(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/video.mp4"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        r = opener.open(urllib.request.Request(url, headers={"Range": "bytes=1000-1009"}))
        assert r.status == 206
        assert r.headers["Content-Range"] == f"bytes 1000-1009/{len(data)}"
        assert r.read() == data[1000:1010]
        full = opener.open(url)
        assert full.status == 200 and full.headers["Accept-Ranges"] == "bytes"
        assert full.read() == data
    finally:
        srv.shutdown()
