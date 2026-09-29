"""Small Telegram transport; callers own credentials, inbound cursors and policy.

Keep the send database outside the vault, in the application's private state
directory. A successful API response proves acceptance, not phone delivery.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.request


class TelegramError(Exception):
    """A safe-to-log failure, without URLs, tokens, chat IDs or server text."""

    def __init__(self, outcome, error_code=None, retry_after=None):
        self.outcome = outcome
        self.error_code = error_code
        self.retry_after = retry_after
        super().__init__(f"Telegram {outcome}; code={error_code}; retry_after={retry_after}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _integer(value):
    return value if type(value) is int else None


class TelegramAPI:
    """HTTPS JSON POSTs to the official API, with no automatic retries."""

    _METHODS = {"getMe", "getWebhookInfo", "getUpdates", "getFile", "sendMessage", "setMyCommands"}

    def __init__(self, token, *, opener=None):
        if not isinstance(token, str) or not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token):
            raise ValueError("Invalid Telegram token format")
        self._token = token
        self._open = opener or urllib.request.build_opener(_NoRedirect()).open

    def post(self, method, payload=None, *, timeout=35):
        if method not in self._METHODS:
            raise ValueError("Unsupported Telegram method")
        if not 1 <= timeout <= 35:
            raise ValueError("HTTP timeout must be between 1 and 35 seconds")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self._token}/{method}",
            data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        http_code = None
        try:
            with self._open(request, timeout=timeout) as response:
                raw = response.read(2_000_001)
        except urllib.error.HTTPError as error:
            http_code = error.code
            try:
                raw = error.read(2_000_001)
            except Exception:
                raw = b""
            finally:
                error.close()
        except Exception:
            # A timeout/reset can occur after sendMessage has been accepted.
            raise TelegramError("ambiguous") from None
        try:
            packet = json.loads(raw) if len(raw) <= 2_000_000 else None
        except (ValueError, UnicodeError):
            packet = None
        if isinstance(packet, dict) and packet.get("ok") is False:
            params = packet.get("parameters")
            retry = _integer(params.get("retry_after")) if isinstance(params, dict) else None
            raise TelegramError(
                "rejected", _integer(packet.get("error_code")) or http_code,
                max(0, retry) if retry is not None else None,
            ) from None
        if http_code is not None:
            outcome = "rejected" if http_code in (400, 401, 403, 404, 409, 429) else "ambiguous"
            raise TelegramError(outcome, http_code) from None
        if not isinstance(packet, dict) or packet.get("ok") is not True or "result" not in packet:
            raise TelegramError("ambiguous") from None
        return packet["result"]

    def get_me(self):
        return self.post("getMe")

    def get_webhook_info(self):
        return self.post("getWebhookInfo")

    def download_file(self, file_id, *, max_bytes=20_000_000):
        """Download a bounded file from Telegram without exposing its secret URL."""
        result = self.post("getFile", {"file_id": file_id})
        path = result.get("file_path") if isinstance(result, dict) else None
        if (not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", path)
                or path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/"))):
            raise TelegramError("rejected")
        size = _integer(result.get("file_size"))
        if size is not None and size > max_bytes:
            raise TelegramError("rejected", 413)
        request = urllib.request.Request(f"https://api.telegram.org/file/bot{self._token}/{path}")
        try:
            with self._open(request, timeout=35) as response:
                data = response.read(max_bytes + 1)
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            raise TelegramError("rejected", code) from None
        except Exception:
            raise TelegramError("ambiguous") from None
        if not data or len(data) > max_bytes:
            raise TelegramError("rejected", 413 if data else None)
        return data

    def get_updates(self, offset=None, timeout=25):
        """Caller must durably handle each update before advancing its offset."""
        if type(timeout) is not int or not 0 <= timeout <= 25:
            raise ValueError("Polling timeout must be an integer from 0 to 25")
        payload = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            if type(offset) is not int or offset < 0:
                raise ValueError("Offset must be a nonnegative integer")
            payload["offset"] = offset
        updates = self.post("getUpdates", payload, timeout=timeout + 10)
        if not isinstance(updates, list) or any(
            not isinstance(item, dict) or _integer(item.get("update_id")) is None
            for item in updates
        ):
            raise TelegramError("ambiguous")
        return updates

    def send_message(self, chat_id, text, reply_to_message_id=None):
        payload = {"chat_id": chat_id, "text": text,
                   "link_preview_options": {"is_disabled": True}}
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": reply_to_message_id}
        return self.post("sendMessage", payload)

    def set_my_commands(self, commands):
        return self.post("setMyCommands", {"commands": commands})


@dataclass(frozen=True)
class SendReceipt:
    status: str
    duplicate: bool = False
    message_id: int | None = None
    error_code: int | None = None
    retry_after: int | None = None


class SendOnce:
    """Bind a chat and durably suppress repeat sends for the same stable key.

    The key identifies one logical message, e.g. morning:2026-01-01. Reusing a
    key with different contents raises ValueError. Rejected attempts require
    retry_rejected=True, and respect retry_after. Ambiguous attempts (including
    process crashes after claiming) remain blocked for manual investigation.
    """

    def __init__(self, api, db_path, chat_id):
        if type(chat_id) is not int or chat_id <= 0:
            raise ValueError("A bound private-chat ID is required")
        self.api = api
        self._chat_id = chat_id
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS sends (
                key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, message_id INTEGER,
                error_code INTEGER, retry_after INTEGER, retry_at REAL
            )""")
        os.chmod(self.db_path, 0o600)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _receipt(row, duplicate=False):
        return SendReceipt(
            row["status"], duplicate, row["message_id"],
            row["error_code"], row["retry_after"],
        )

    def send(self, key, text, reply_to_message_id=None, *, retry_rejected=False):
        if not isinstance(key, str) or not key:
            raise ValueError("A stable nonempty send key is required")
        if not isinstance(text, str) or not text:
            raise ValueError("Message text must be nonempty")
        content = json.dumps([self._chat_id, text, reply_to_message_id], ensure_ascii=False)
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM sends WHERE key = ?", (key,)).fetchone()
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise ValueError("Send key was already used with different contents")
                if (previous["status"] != "rejected" or not retry_rejected
                        or (previous["retry_at"] or 0) > time.time()):
                    return self._receipt(previous, duplicate=True)
                db.execute("UPDATE sends SET status = 'ambiguous', error_code = NULL, "
                           "retry_after = NULL, retry_at = NULL WHERE key = ?", (key,))
            else:
                db.execute("INSERT INTO sends (key, fingerprint, status) VALUES (?, ?, 'ambiguous')",
                           (key, fingerprint))
        # The claim is committed before any request. A crash cannot trigger a
        # blind resend, even if it happens after acceptance but before storage.
        try:
            result = self.api.send_message(self._chat_id, text, reply_to_message_id)
            message_id = _integer(result.get("message_id")) if isinstance(result, dict) else None
            if message_id is None:
                raise TelegramError("ambiguous")
            receipt = SendReceipt("sent", message_id=message_id)
        except TelegramError as error:
            receipt = SendReceipt(error.outcome, error_code=error.error_code,
                                  retry_after=error.retry_after)
        except Exception:
            # Keep unexpected transport exceptions safe to log, too.
            receipt = SendReceipt("ambiguous")
        retry_at = time.time() + receipt.retry_after if receipt.retry_after else None
        with self._connect() as db:
            db.execute("UPDATE sends SET status = ?, message_id = ?, error_code = ?, "
                       "retry_after = ?, retry_at = ? WHERE key = ?",
                       (receipt.status, receipt.message_id, receipt.error_code,
                        receipt.retry_after, retry_at, key))
        return receipt
