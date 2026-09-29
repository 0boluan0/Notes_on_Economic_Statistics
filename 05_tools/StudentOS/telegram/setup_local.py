"""One-use localhost form for storing a BotFather token outside the vault."""

import argparse
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
from urllib.parse import parse_qs

from telegram_api import TelegramAPI, TelegramError


class SetupError(Exception):
    """Only fixed, safe-to-display messages belong here."""


def external_state_dir(state_dir, vault):
    state_dir = Path(state_dir).expanduser().resolve()
    vault = Path(vault).expanduser().resolve()
    if state_dir == vault or vault in state_dir.parents:
        raise SetupError("The credentials directory must be outside the vault.")
    return state_dir


class SetupSession:
    def __init__(self, state_dir, vault, *, api_factory=TelegramAPI):
        self.state_dir = external_state_dir(state_dir, vault)
        self.nonce = secrets.token_urlsafe(32)
        self.api_factory = api_factory
        self.status = {"configured": False}
        if (self.state_dir / "config.json").exists():
            raise SetupError("A configuration already exists; it will not be overwritten.")

    def submit(self, nonce, token):
        if not isinstance(nonce, str) or not secrets.compare_digest(nonce, self.nonce):
            raise SetupError("Invalid setup nonce.")
        if self.status["configured"]:
            raise SetupError("This setup form has already been used.")
        try:
            api = self.api_factory(token.strip())
            bot = api.get_me()
            webhook = api.get_webhook_info()
        except (TelegramError, ValueError):
            raise SetupError("Token validation failed. Check the token and try again.") from None
        if (not isinstance(bot, dict) or bot.get("is_bot") is not True
                or not isinstance(bot.get("username"), str)
                or not re.fullmatch(r"[A-Za-z0-9_]{5,32}", bot["username"])):
            raise SetupError("The API did not return a valid bot identity.")
        if not isinstance(webhook, dict) or not isinstance(webhook.get("url"), str):
            raise SetupError("The webhook state could not be verified.")
        if webhook["url"]:
            raise SetupError("This bot already has a webhook. Use a new bot; no webhook was changed.")
        pairing_code = secrets.token_urlsafe(24)
        config = {"token": token.strip(), "bot_username": bot["username"],
                  "pairing_code": pairing_code, "owner_chat_id": None}
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.state_dir,
                                             prefix=".setup-", delete=False) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                json.dump(config, stream, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Publish a complete file atomically without replacing any existing
            # file (including a symlink created while validation was running).
            os.link(temporary, self.state_dir / "config.json")
        except OSError:
            raise SetupError("Could not save credentials securely; no existing configuration was replaced.") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.status = {"configured": True, "bot_username": bot["username"],
                       "pairing_url": f"https://t.me/{bot['username']}?start={pairing_code}"}
        return dict(self.status)


def setup_page(session, error=None):
    if session.status["configured"]:
        pairing_url = html.escape(session.status["pairing_url"], quote=True)
        content = ("<h1>机器人已验证</h1><p>凭据已保存在这台电脑的私有目录中，服务配置正在进行。</p>"
                   f'<p><a href="{pairing_url}" rel="noreferrer">打开 Telegram，点击 Start 完成配对</a></p>'
                   "<p>配对完成后，接入程序会继续验证消息收发。</p>")
    else:
        note = f'<p role="alert">{html.escape(error)}</p>' if error else ""
        content = ("<h1>连接 Student OS</h1>"
                   "<p>在 Telegram 的官方 @BotFather 创建机器人，将它给你的 token 粘贴到下方。</p>"
                   "<p>token 只会从本机发往 Telegram 官方接口进行验证，并保存在本机私有目录中。</p>"
                   f'{note}<form method="post" action="/{session.nonce}" autocomplete="off">'
                   f'<input type="hidden" name="nonce" value="{session.nonce}">'
                   '<label for="token">BotFather token</label>'
                   '<input id="token" name="token" type="password" required maxlength="512" '
                   'autocomplete="new-password" spellcheck="false">'
                   '<button type="submit">验证并保存</button></form>')
    return ("<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Student OS · Telegram</title><style>"
            "body{font:18px system-ui;max-width:620px;margin:70px auto;padding:0 24px;line-height:1.6}"
            "input,button{display:block;box-sizing:border-box;width:100%;padding:12px;margin:12px 0}"
            "button{cursor:pointer}a{color:#126bb8}[role=alert]{color:#a21c1c}"
            f"</style><body>{content}</body></html>").encode("utf-8")


class SetupHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _allowed_origin(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        return (self.headers.get("Host") == expected
                and self.headers.get("Origin", f"http://{expected}") == f"http://{expected}"
                and self.headers.get("Sec-Fetch-Site") != "cross-site")

    def _reply(self, code, body, content_type="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Preserve same-origin POST Origin; external pairing links use rel=noreferrer.
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; "
                         "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        session = self.server.session
        if not self._allowed_origin():
            return self._reply(403, b"Forbidden")
        if self.path == f"/{session.nonce}/status":
            return self._reply(200, json.dumps(session.status).encode(), "application/json")
        if self.path != f"/{session.nonce}":
            return self._reply(404, b"Not found")
        self._reply(200, setup_page(session))

    def do_POST(self):
        session = self.server.session
        if not self._allowed_origin() or self.path != f"/{session.nonce}":
            return self._reply(403, b"Forbidden")
        if session.status["configured"]:
            return self._reply(409, b"Setup already completed")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (not 0 < length <= 4096 or self.headers.get("Transfer-Encoding")
                    or self.headers.get_content_type() != "application/x-www-form-urlencoded"):
                return self._reply(400, b"Invalid form")
            values = parse_qs(self.rfile.read(length).decode("utf-8"), strict_parsing=True)
            if set(values) != {"nonce", "token"} or any(len(v) != 1 for v in values.values()):
                return self._reply(400, b"Invalid form")
            status = session.submit(values["nonce"][0], values["token"][0])
        except SetupError as error:
            return self._reply(400, setup_page(session, str(error)))
        except (ValueError, UnicodeError):
            return self._reply(400, b"Invalid form")
        except Exception:
            return self._reply(500, b"Setup failed; no diagnostic contains your token.")
        # Never print config or token; this is the complete permitted output.
        print(json.dumps(status), flush=True)
        self._reply(200, setup_page(session))


class SetupServer(HTTPServer):
    def handle_error(self, request, client_address):
        # Default HTTPServer diagnostics include tracebacks. Keep this quiet.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True)
    parser.add_argument("--state-dir", type=Path,
                        default=Path.home() / "Library/Application Support/StudentOSTelegram")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    try:
        session = SetupSession(args.state_dir, args.vault)
        server = SetupServer(("127.0.0.1", args.port), SetupHandler)
    except SetupError as error:
        parser.exit(1, f"{error}\n")
    except OSError:
        parser.exit(1, "Could not start the localhost setup server.\n")
    server.session = session
    print(f"http://127.0.0.1:{server.server_port}/{session.nonce}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
