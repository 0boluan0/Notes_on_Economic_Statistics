"""Offline protocol and crash/deduplication checks. Never contacts Telegram."""

import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error

from telegram_api import SendOnce, TelegramAPI, TelegramError, _NoRedirect


class TelegramAPITests(unittest.TestCase):
    def api(self, packet):
        opener = Mock(return_value=io.BytesIO(json.dumps(packet).encode()))
        return TelegramAPI("123:fake_secret", opener=opener), opener

    def test_updates_preserve_ids_and_explicit_cursor(self):
        updates = [{"update_id": 9, "message": {"text": "test"}}]
        api, opener = self.api({"ok": True, "result": updates})
        self.assertEqual(api.get_updates(offset=8, timeout=25), updates)
        request = opener.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data),
                         {"offset": 8, "timeout": 25, "allowed_updates": ["message"]})
        self.assertEqual(opener.call_args.kwargs["timeout"], 35)
        self.assertEqual(request.full_url, "https://api.telegram.org/bot123:fake_secret/getUpdates")

    def test_poll_limits_cannot_silently_discard_updates(self):
        api, opener = self.api({"ok": True, "result": []})
        for arguments in ({"offset": -1}, {"timeout": 26}, {"offset": True}):
            with self.assertRaises(ValueError):
                api.get_updates(**arguments)
        opener.assert_not_called()

    def test_rejection_preserves_code_and_retry_without_description(self):
        api, _ = self.api({"ok": False, "error_code": 429,
                          "description": "URL /bot123:fake_secret private chat 998877",
                          "parameters": {"retry_after": 20}})
        with self.assertRaises(TelegramError) as error:
            api.send_message(998877, "hello")
        self.assertEqual(error.exception.outcome, "rejected")
        self.assertEqual(error.exception.error_code, 429)
        self.assertEqual(error.exception.retry_after, 20)
        self.assertNotIn("fake_secret", str(error.exception))
        self.assertNotIn("998877", str(error.exception))

    def test_network_errors_are_ambiguous_and_sanitized(self):
        opener = Mock(side_effect=urllib.error.URLError("https://x/bot123:fake_secret"))
        api = TelegramAPI("123:fake_secret", opener=opener)
        with self.assertRaises(TelegramError) as error:
            api.send_message(998877, "hello")
        self.assertEqual(error.exception.outcome, "ambiguous")
        self.assertNotIn("fake_secret", str(error.exception))

    def test_http_error_json_rejection(self):
        body = io.BytesIO(b'{"ok":false,"error_code":403}')
        opener = Mock(side_effect=urllib.error.HTTPError("secret", 403, "secret", {}, body))
        api = TelegramAPI("123:fake_secret", opener=opener)
        with self.assertRaises(TelegramError) as error:
            api.get_me()
        self.assertEqual(error.exception.outcome, "rejected")
        self.assertEqual(error.exception.error_code, 403)

    def test_invalid_server_response_is_ambiguous(self):
        api, _ = self.api({"ok": True})
        with self.assertRaises(TelegramError) as error:
            api.get_me()
        self.assertEqual(error.exception.outcome, "ambiguous")

    def test_arbitrary_methods_and_redirects_are_blocked(self):
        api, opener = self.api({"ok": True, "result": True})
        with self.assertRaises(ValueError):
            api.post("../../other")
        opener.assert_not_called()
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://other"))

    def test_file_download_is_bounded_and_uses_official_endpoint(self):
        opener = Mock(side_effect=[io.BytesIO(b'{"ok":true,"result":{"file_path":"voice/file_1.oga"}}'), io.BytesIO(b'audio')])
        api = TelegramAPI('123:fake_secret', opener=opener)
        self.assertEqual(api.download_file('voice-id', max_bytes=5), b'audio')
        self.assertEqual(json.loads(opener.call_args_list[0].args[0].data), {'file_id': 'voice-id'})
        self.assertEqual(opener.call_args.args[0].full_url, 'https://api.telegram.org/file/bot123:fake_secret/voice/file_1.oga')
        opener.side_effect = [io.BytesIO(b'{"ok":true,"result":{"file_path":"voice/file_1.oga"}}'), io.BytesIO(b'too large')]
        with self.assertRaises(TelegramError) as error:
            api.download_file('voice-id', max_bytes=5)
        self.assertEqual(error.exception.error_code, 413)

    def test_file_download_blocks_path_injection_and_sanitizes_failures(self):
        for path in ('../x', '/absolute', 'https://other/x', 'voice/../../x'):
            api, opener = self.api({'ok': True, 'result': {'file_path': path}})
            with self.assertRaises(TelegramError):
                api.download_file('voice-id')
            self.assertEqual(opener.call_count, 1)
        opener = Mock(side_effect=[io.BytesIO(b'{"ok":true,"result":{"file_path":"voice/file.oga"}}'),
                                   urllib.error.URLError('secret URL: fake_secret')])
        with self.assertRaises(TelegramError) as error:
            TelegramAPI('123:fake_secret', opener=opener).download_file('voice-id')
        self.assertNotIn('fake_secret', str(error.exception))


class SendOnceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sends.sqlite3"
        self.api = Mock()
        self.api.send_message.return_value = {"message_id": 17}
        self.sender = SendOnce(self.api, self.path, 998877)

    def test_persistent_dedup_and_no_private_message_storage(self):
        first = self.sender.send("morning:date", "private message")
        second = SendOnce(self.api, self.path, 998877).send("morning:date", "private message")
        self.assertEqual((first.status, first.duplicate), ("sent", False))
        self.assertEqual((second.status, second.duplicate, second.message_id), ("sent", True, 17))
        self.api.send_message.assert_called_once_with(998877, "private message", None)
        raw = self.path.read_bytes()
        self.assertNotIn(b"private message", raw)
        self.assertNotIn(b"998877", raw)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_same_key_different_contents_fails(self):
        self.sender.send("key", "original")
        with self.assertRaises(ValueError):
            self.sender.send("key", "changed")
        self.assertEqual(self.api.send_message.call_count, 1)

    def test_ambiguous_never_blindly_retries(self):
        self.api.send_message.side_effect = TelegramError("ambiguous")
        self.assertEqual(self.sender.send("key", "hello").status, "ambiguous")
        self.assertTrue(self.sender.send("key", "hello", retry_rejected=True).duplicate)
        self.assertEqual(self.api.send_message.call_count, 1)

    def test_rejected_requires_explicit_retry(self):
        self.api.send_message.side_effect = [TelegramError("rejected", 403), {"message_id": 18}]
        self.assertEqual(self.sender.send("key", "hello").status, "rejected")
        self.assertTrue(self.sender.send("key", "hello").duplicate)
        self.assertEqual(self.sender.send("key", "hello", retry_rejected=True).status, "sent")
        self.assertEqual(self.api.send_message.call_count, 2)

    def test_explicit_retry_still_respects_server_cooldown(self):
        self.api.send_message.side_effect = [TelegramError("rejected", 429, 30), {"message_id": 19}]
        with patch("telegram_api.time.time", return_value=100):
            self.sender.send("key", "hello")
            self.assertTrue(self.sender.send("key", "hello", retry_rejected=True).duplicate)
        with patch("telegram_api.time.time", return_value=131):
            self.assertEqual(self.sender.send("key", "hello", retry_rejected=True).status, "sent")
        self.assertEqual(self.api.send_message.call_count, 2)

    def test_claim_is_durable_before_request_and_survives_crash(self):
        def simulate_crash(*args):
            with sqlite3.connect(self.path) as db:
                self.assertEqual(db.execute("SELECT status FROM sends").fetchone()[0], "ambiguous")
            raise KeyboardInterrupt()
        self.api.send_message.side_effect = simulate_crash
        with self.assertRaises(KeyboardInterrupt):
            self.sender.send("key", "hello")
        self.assertTrue(SendOnce(self.api, self.path, 998877).send("key", "hello").duplicate)
        self.assertEqual(self.api.send_message.call_count, 1)

    def test_concurrent_claim_makes_only_one_request(self):
        entered = threading.Event()
        release = threading.Event()
        def blocked_send(*args):
            entered.set()
            release.wait(3)
            return {"message_id": 20}
        self.api.send_message.side_effect = blocked_send
        worker = threading.Thread(target=self.sender.send, args=("key", "hello"))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            result = SendOnce(self.api, self.path, 998877).send("key", "hello")
            self.assertEqual((result.status, result.duplicate), ("ambiguous", True))
        finally:
            release.set()
            worker.join(3)
        self.assertEqual(self.api.send_message.call_count, 1)


if __name__ == "__main__":
    unittest.main()
