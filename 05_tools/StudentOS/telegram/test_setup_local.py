"""Offline onboarding checks: fake API and fake sockets only."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.parse import urlencode

from setup_local import SetupError, SetupHandler, SetupSession, external_state_dir, setup_page


class FakeSocket:
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.response = b""

    def makefile(self, *args):
        return self.request

    def sendall(self, data):
        self.response += data


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.vault = self.base / "vault"
        self.vault.mkdir()
        self.state = self.base / "private"
        self.api = Mock()
        self.api.get_me.return_value = {"is_bot": True, "username": "StudyExampleBot"}
        self.api.get_webhook_info.return_value = {"url": ""}
        self.factory = Mock(return_value=self.api)
        self.session = SetupSession(self.state, self.vault, api_factory=self.factory)

    def test_inside_vault_and_symlink_paths_are_rejected(self):
        for path in (self.vault, self.vault / "subdir"):
            with self.assertRaises(SetupError):
                external_state_dir(path, self.vault)
        (self.base / "alias").symlink_to(self.vault, target_is_directory=True)
        with self.assertRaises(SetupError):
            external_state_dir(self.base / "alias" / "state", self.vault)

    def test_nonce_mismatch_does_not_touch_api_or_disk(self):
        with self.assertRaises(SetupError):
            self.session.submit("incorrect", "123:secret")
        self.factory.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_valid_submit_saves_private_config_and_consumes_form(self):
        status = self.session.submit(self.session.nonce, "123:secret")
        config = self.state / "config.json"
        saved = json.loads(config.read_text())
        self.assertEqual(saved["token"], "123:secret")
        self.assertEqual(set(saved), {"token", "bot_username", "pairing_code", "owner_chat_id"})
        self.assertIsNone(saved["owner_chat_id"])
        self.assertEqual(config.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)
        self.assertNotIn("secret", json.dumps(status))
        self.assertNotIn(b"secret", setup_page(self.session))
        self.assertEqual(status["pairing_url"],
                         f"https://t.me/StudyExampleBot?start={saved['pairing_code']}")
        with self.assertRaises(SetupError):
            self.session.submit(self.session.nonce, "123:changed")
        self.assertEqual(json.loads(config.read_text())["token"], "123:secret")

    def test_existing_webhook_is_rejected_without_changes(self):
        self.api.get_webhook_info.return_value = {"url": "https://example.org/private"}
        with self.assertRaises(SetupError):
            self.session.submit(self.session.nonce, "123:secret")
        self.assertFalse(self.state.exists())
        self.assertEqual([call[0] for call in self.api.method_calls], ["get_me", "get_webhook_info"])

    def test_existing_configuration_is_never_overwritten(self):
        self.state.mkdir()
        config = self.state / "config.json"
        config.write_text("existing")
        with self.assertRaises(SetupError):
            self.session.submit(self.session.nonce, "123:secret")
        self.assertEqual(config.read_text(), "existing")
        self.assertEqual(list(self.state.iterdir()), [config])

    def request(self, body, origin="http://127.0.0.1:12345"):
        raw = (f"POST /{self.session.nonce} HTTP/1.1\r\nHost: 127.0.0.1:12345\r\n"
               f"Origin: {origin}\r\nContent-Type: application/x-www-form-urlencoded\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        socket = FakeSocket(raw)
        server = Mock(session=self.session, server_port=12345)
        output = io.StringIO()
        with redirect_stdout(output):
            SetupHandler(socket, ("127.0.0.1", 23456), server)
        return socket.response, output.getvalue()

    def test_http_nonce_and_origin_guards(self):
        response, output = self.request(urlencode({"nonce": "wrong", "token": "123:secret"}).encode())
        self.assertIn(b"400", response.splitlines()[0])
        self.assertNotIn(b"secret", response)
        self.assertEqual(output, "")
        self.factory.assert_not_called()
        response, _ = self.request(urlencode({"nonce": self.session.nonce, "token": "123:secret"}).encode(),
                                   origin="https://external.example")
        self.assertIn(b"403", response.splitlines()[0])
        self.factory.assert_not_called()

    def test_http_success_has_security_headers_and_safe_output(self):
        response, output = self.request(urlencode({"nonce": self.session.nonce, "token": "123:secret"}).encode())
        self.assertIn(b"200", response.splitlines()[0])
        self.assertIn(b"Content-Security-Policy: default-src 'none'", response)
        self.assertIn(b"Cache-Control: no-store", response)
        self.assertNotIn(b"123:secret", response)
        self.assertNotIn("123:secret", output)
        self.assertEqual(json.loads(output)["bot_username"], "StudyExampleBot")


if __name__ == "__main__":
    unittest.main()
