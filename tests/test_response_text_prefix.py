import unittest
from unittest.mock import PropertyMock, patch

import requests

from autoresttest.utils import get_response_text_prefix


def make_response(content, content_type=None, encoding=None):
    response = requests.Response()
    response.status_code = 200
    response._content = content
    if content_type:
        response.headers["Content-Type"] = content_type
    response.encoding = encoding
    return response


class ResponseTextPrefixTests(unittest.TestCase):
    def test_binary_bodies_are_not_scanned_for_an_encoding(self):
        response = make_response(b"\xa1\xb6" * 2_000_000, "application/octet-stream")
        with patch.object(
            requests.Response,
            "apparent_encoding",
            new_callable=PropertyMock,
            side_effect=AssertionError("full-body encoding detection"),
        ):
            text = get_response_text_prefix(response)
        self.assertLessEqual(len(text), 1000)

    def test_text_bodies_match_the_start_of_response_text(self):
        body = '{"message": "email must be valid", "field": "email"}'
        response = make_response(body.encode(), "application/json", "utf-8")
        self.assertEqual(get_response_text_prefix(response), body)
        self.assertEqual(get_response_text_prefix(response, max_bytes=10), body[:10])

    def test_declared_charset_is_used(self):
        response = make_response("café".encode("latin-1"), encoding="ISO-8859-1")
        self.assertEqual(get_response_text_prefix(response), "café")

    def test_unknown_charset_and_missing_body_do_not_raise(self):
        self.assertEqual(
            get_response_text_prefix(make_response(b"ok", encoding="not-a-codec")),
            "ok",
        )
        self.assertEqual(get_response_text_prefix(make_response(None)), "")


if __name__ == "__main__":
    unittest.main()
