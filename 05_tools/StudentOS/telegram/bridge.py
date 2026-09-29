#!/usr/bin/env python3
"""Private, one-owner Telegram inbox and idempotent Student OS notifications."""
import argparse
import fcntl
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import tomllib

from telegram_api import TelegramAPI, TelegramError, SendOnce
from voice import transcribe_message, VoiceError


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def private_state(path, vault):
    vault = Path(vault).expanduser().resolve()
    path = Path(path).expanduser().resolve()
    if path == vault or vault in path.parents:
        raise ValueError('State must be outside the vault')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


class Inbox:
    def __init__(self, state):
        self.db = sqlite3.connect(state / 'inbox.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY, offset INTEGER)')
        self.db.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, payload TEXT, reply TEXT, status TEXT, attempts INTEGER DEFAULT 0)')
        os.chmod(state / 'inbox.sqlite3', 0o600)

    def offset(self):
        row = self.db.execute('SELECT offset FROM cursor WHERE id=1').fetchone()
        return row[0] if row else None

    def receive(self, updates, config):
        # Commit the original authorized message before acknowledging its offset.
        with self.db:
            for update in updates:
                message = update.get('message', {})
                chat = message.get('chat', {})
                sender = message.get('from', {})
                text = message.get('text', '')
                private = (chat.get('type') == 'private' and type(chat.get('id')) is int
                           and chat['id'] > 0 and sender.get('id') == chat['id'])
                owner = config.get('owner_chat_id')
                pairing = (owner is None and isinstance(text, str)
                           and hmac.compare_digest(text.encode(), ('/start ' + config['pairing_code']).encode()))
                if private and (chat['id'] == owner or pairing):
                    self.db.execute('INSERT OR IGNORE INTO messages (id,payload,status) VALUES (?,?,?)',
                                    (update['update_id'], json.dumps(message, ensure_ascii=False), 'pending'))
            if updates:
                next_offset = max(item['update_id'] for item in updates) + 1
                self.db.execute('INSERT INTO cursor VALUES (1,?) ON CONFLICT(id) DO UPDATE SET offset=max(offset,excluded.offset)', (next_offset,))

    def pending(self):
        return self.db.execute("SELECT * FROM messages WHERE status='pending' ORDER BY id").fetchall()

    def reply(self, update_id, text):
        with self.db:
            self.db.execute('UPDATE messages SET reply=? WHERE id=?', (text, update_id))

    def finish(self, update_id, status):
        with self.db:
            self.db.execute('UPDATE messages SET status=? WHERE id=?', (status, update_id))


HELP = ('已连接 Student OS。可以打字，也可以直接发语音，比如“Python 前两节做完了”或“今天只完成了笔记”。'
        '我会回复实际记下了什么；不清楚的地方再确认。\n'
        '/today 查看最近一次晨间安排\n/status 查看服务状态')


def schedule_description(path=None):
    path = path or Path.home() / '.codex/automations/student-os-today/automation.toml'
    try:
        task = tomllib.loads(Path(path).read_text())
        if task.get('status') != 'ACTIVE':
            return '原 Student OS 定时任务目前已暂停。'
        rule = task.get('rrule', '')
        hours = re.search(r'(?:^|;)BYHOUR=([0-9,]+)(?:;|$)', rule)
        minute = re.search(r'(?:^|;)BYMINUTE=(\d+)(?:;|$)', rule)
        if 'FREQ=DAILY' in rule and hours and minute:
            values = sorted(set(int(hour) for hour in hours[1].split(',')))
            if values and all(0 <= hour <= 23 for hour in values) and 0 <= int(minute[1]) < 60:
                times = '、'.join(f'{hour:02d}:{int(minute[1]):02d}' for hour in values)
                return f'原 Student OS 任务当前每天 {times} 运行（伦敦时间），始终复用同一个任务。'
    except (OSError, ValueError, TypeError):
        pass
    return '晨报和晚间对账沿用原 Student OS 任务中的最新时间，并复用同一个任务。'


def handle(inbox, row, config, api, state, vault, codex):
    message = json.loads(row['payload'])
    owner = config.get('owner_chat_id')
    if owner is not None and message['chat']['id'] != owner:
        inbox.finish(row['id'], 'ignored')
        return
    if owner is None:
        config['owner_chat_id'] = message['chat']['id']
        save_json(state / 'config.json', config)
    text = message.get('text', '')
    reply = row['reply']
    if reply is None:
        if text.startswith('/start') or text == '/help':
            reply = HELP
        elif text == '/status':
            reply = '收信服务正在运行。你发来的进展会记入私有对账记录。\n' + schedule_description()
        elif text == '/today':
            latest = state / 'latest-morning.json'
            if latest.exists():
                data = json.loads(latest.read_text())
                reply = data['date'] + '\n' + data['text']
            else:
                reply = '还没有生成 Telegram 晨间安排。接通后的下次 06:00 会推送；你也可以直接回复最近的学习进展。'
        elif message.get('voice') or message.get('audio'):
            try:
                transcript = transcribe_message(api, message, state, row['id'])
                from reconciliation import process_reply
                result, _report = process_reply(vault, '【语音转写】' + transcript, codex, state, row['id'])
                heard = transcript if len(transcript) <= 400 else transcript[:400] + '…'
                reply = '听到的是：「' + heard + '」\n\n' + result
            except VoiceError as error:
                reply = str(error)
        elif not text or len(text) > 6000:
            reply = '可以直接发文字或语音，告诉我做了什么。文字请分成较短的几条发送。'
        else:
            from reconciliation import process_reply
            reply, _report = process_reply(vault, text, codex, state, row['id'])
        inbox.reply(row['id'], reply)
    receipt = SendOnce(api, state / 'sends.sqlite3', config['owner_chat_id']).send(
        f"reply:{row['id']}", reply, message['message_id'], retry_rejected=True)
    if receipt.status == 'rejected' and receipt.error_code == 429:
        return  # A durable reply remains pending; SendOnce enforces Telegram's cooldown.
    inbox.finish(row['id'], receipt.status)


def serve(args, state, vault, config):
    lock = open(state / 'service.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = TelegramAPI(config['token'])
    if api.get_webhook_info().get('url'):
        raise ValueError('Existing webhook must be reviewed before polling')
    inbox = Inbox(state)
    delay = 2
    while True:
        try:
            for row in inbox.pending():
                try:
                    handle(inbox, row, config, api, state, vault, args.codex)
                except Exception as error:
                    # Do not emit exception text: it may contain user data.
                    save_json(state / 'health.json', {'last_poll': time.time(), 'status': 'processing_error', 'error_type': type(error).__name__})
                    with inbox.db:
                        inbox.db.execute('UPDATE messages SET attempts=attempts+1 WHERE id=?', (row['id'],))
                    if row['attempts'] >= 2:
                        fallback = '这条进展已保存在本机收件记录，但自动对账暂时失败，尚未确认更新任务。请先继续你的事；需要在 Student OS 中检查这条记录。'
                        cached = inbox.db.execute('SELECT reply FROM messages WHERE id=?', (row['id'],)).fetchone()[0]
                        if cached is not None:
                            fallback = cached  # Keep an existing outbound fingerprint unchanged.
                        else:
                            inbox.reply(row['id'], fallback)
                        receipt = SendOnce(api, state / 'sends.sqlite3', config['owner_chat_id']).send(
                            f"reply:{row['id']}", fallback, json.loads(row['payload'])['message_id'])
                        inbox.finish(row['id'], 'processing_failed_' + receipt.status)
                    # Hold this item for the next loop; its durable receipt prevents duplicate writes.
                    break
            inbox.receive(api.get_updates(inbox.offset(), timeout=25), config)
            save_json(state / 'health.json', {'last_poll': time.time(), 'status': 'ok'})
            delay = 2
        except Exception as error:
            save_json(state / 'health.json', {'last_error': time.time(), 'status': 'poll_error', 'error_type': type(error).__name__})
            time.sleep(delay)
            delay = min(delay * 2, 60)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vault', required=True)
    parser.add_argument('--state-dir', default='~/Library/Application Support/StudentOSTelegram')
    parser.add_argument('--codex', default='/opt/homebrew/bin/codex')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('serve')
    commands.add_parser('status')
    send = commands.add_parser('send')
    send.add_argument('--kind', choices=('morning', 'evening', 'test'), required=True)
    send.add_argument('--date', required=True)
    send.add_argument('--text-file', default='-', help='UTF-8 file, or stdin')
    send.add_argument('--retry-rejected', action='store_true', help='Explicitly retry a proven rejection with identical contents')
    args = parser.parse_args()
    vault = Path(args.vault).expanduser().resolve()
    state = private_state(args.state_dir, vault)
    config_path = state / 'config.json'
    if args.command == 'status':
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        health = json.loads((state / 'health.json').read_text()) if (state / 'health.json').exists() else {}
        print(json.dumps({'configured': bool(config), 'bot_username': config.get('bot_username'), 'paired': bool(config.get('owner_chat_id')), 'schedule': schedule_description(), 'health': health}, ensure_ascii=False))
        return
    config = json.loads(config_path.read_text())
    if args.command == 'serve':
        serve(args, state, vault, config)
        return
    if not config.get('owner_chat_id'):
        raise ValueError('Open the pairing link and press Start first')
    text = sys.stdin.read() if args.text_file == '-' else Path(args.text_file).read_text(encoding='utf-8')
    text = text.strip()
    if not text or len(text) > 3500:
        raise ValueError('Notification must have 1 to 3500 characters')
    receipt = SendOnce(TelegramAPI(config['token']), state / 'sends.sqlite3', config['owner_chat_id']).send(f'{args.kind}:{args.date}', text, retry_rejected=args.retry_rejected)
    if receipt.status == 'sent' and args.kind in ('morning', 'evening'):
        save_json(state / ('latest-' + args.kind + '.json'), {'date': args.date, 'text': text})
    print(json.dumps({'status': receipt.status, 'duplicate': receipt.duplicate, 'error_code': receipt.error_code}))
    if receipt.status != 'sent':
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Student OS Telegram: ' + type(error).__name__, file=sys.stderr)
        sys.exit(1)
