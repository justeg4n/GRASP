import pytest

from grasp.llm.errors import ErrorKind, classify


@pytest.mark.parametrize(
    "status,msg,kind",
    [
        (None, "connection reset", ErrorKind.TRANSIENT),
        (429, "rate limit exceeded", ErrorKind.TRANSIENT),
        (429, "You exceeded your current quota", ErrorKind.CREDIT),
        (500, "internal error", ErrorKind.TRANSIENT),
        (503, "overloaded", ErrorKind.TRANSIENT),
        (402, "", ErrorKind.CREDIT),
        (400, "Insufficient balance", ErrorKind.CREDIT),
        (403, "Your account has insufficient credit", ErrorKind.CREDIT),
        (401, "invalid api key", ErrorKind.AUTH),
        (403, "forbidden", ErrorKind.AUTH),
        (404, "model not found", ErrorKind.CONFIG),
        (400, "prompt is too long", ErrorKind.BAD_REQUEST),
        (418, "teapot", ErrorKind.UNKNOWN_4XX),
    ],
)
def test_classify(status, msg, kind):
    assert classify(status, msg) is kind
