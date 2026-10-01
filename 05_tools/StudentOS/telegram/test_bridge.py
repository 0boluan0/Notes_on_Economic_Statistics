import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bridge import Inbox, handle, private_state, schedule_description
from telegram_api import TelegramError
from voice import VoiceError


def update(number, text='progress', chat=123, kind='private', sender=None):
    return {'update_id': number, 'message': {'message_id': number, 'text': text,
            'chat': {'id': chat, 'type': kind}, 'from': {'id': chat if sender is None else sender}}}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.inbox = Inbox(self.state)
        self.addCleanup(self.inbox.db.close)

    def test_only_owner_private_messages_are_queued(self):
        self.inbox.receive([update(1), update(2, chat=456), update(3, kind='group'),
                            update(4, sender=456)], {'owner_chat_id': 123})
        self.assertEqual([row['id'] for row in self.inbox.pending()], [1])
        self.assertEqual(self.inbox.offset(), 5)

    def test_pairing_requires_exact_nonce(self):
        config = {'owner_chat_id': None, 'pairing_code': 'secret'}
        self.inbox.receive([update(1, '/start'), update(2, '/start wrong'),
                            update(3, '/start secret')], config)
        self.assertEqual([row['id'] for row in self.inbox.pending()], [3])

    def test_reopen_retains_message_before_offset_ack(self):
        self.inbox.receive([update(4)], {'owner_chat_id': 123})
        another = Inbox(self.state)
        self.addCleanup(another.db.close)
        self.assertEqual(another.offset(), 5)
        self.assertEqual(len(another.pending()), 1)
        another.receive([update(4)], {'owner_chat_id': 123})
        self.assertEqual(len(another.pending()), 1)

    def test_owner_binding_and_reply_replay_do_not_repeat_classifier(self):
        class API:
            def send_message(self, *args):
                return {'message_id': 20}
        config = {'owner_chat_id': None, 'pairing_code': 'secret'}
        self.inbox.receive([update(3, '/start secret')], config)
        row = self.inbox.pending()[0]
        handle(self.inbox, row, config, API(), self.state, self.state, 'codex')
        self.assertEqual(json.loads((self.state / 'config.json').read_text())['owner_chat_id'], 123)
        self.assertEqual(len(self.inbox.pending()), 0)
        self.inbox.receive([update(4, 'done')], config)
        self.inbox.reply(4, 'cached result')
        with patch('reconciliation.process_reply', side_effect=AssertionError('must use cache')):
            handle(self.inbox, self.inbox.pending()[0], config, API(), self.state, self.state, 'codex')

    def test_state_cannot_be_inside_vault(self):
        with self.assertRaises(ValueError):
            private_state(self.state / 'secret', self.state)

    def test_status_uses_actual_automation_time(self):
        path = self.state / 'automation.toml'
        path.write_text('status="ACTIVE"\nrrule="FREQ=DAILY;BYHOUR=6,21;BYMINUTE=0"\n')
        self.assertIn('06:00、21:00', schedule_description(path))
        self.assertNotIn('22:00', schedule_description(path))

    def test_rate_limit_keeps_cached_reply_pending_and_honors_cooldown(self):
        class API:
            calls = 0
            def send_message(self, *args):
                self.calls += 1
                raise TelegramError('rejected', 429, 30)
        api = API()
        config = {'owner_chat_id': 123}
        self.inbox.receive([update(6, '/help')], config)
        handle(self.inbox, self.inbox.pending()[0], config, api, self.state, self.state, 'codex')
        self.assertEqual(len(self.inbox.pending()), 1)
        self.assertTrue(self.inbox.pending()[0]['reply'])
        handle(self.inbox, self.inbox.pending()[0], config, api, self.state, self.state, 'codex')
        self.assertEqual(api.calls, 1)

    def test_voice_is_transcribed_then_reconciled_and_replay_is_cached(self):
        class API:
            def send_message(self, *args):
                return {'message_id': 30}
        config = {'owner_chat_id': 123}
        event = update(9)
        del event['message']['text']
        event['message']['voice'] = {'file_id': 'voice-id', 'duration': 12, 'file_size': 50000}
        self.inbox.receive([event], config)
        with patch('bridge.transcribe_message', create=True, return_value='收到测试') as transcribe, \
             patch('reconciliation.process_reply', return_value=('已记下你的回复，暂未勾选任务。', {})) as reconcile:
            handle(self.inbox, self.inbox.pending()[0], config, API(), self.state, self.state, 'codex')
            transcribe.assert_called_once()
            self.assertIn('收到测试', reconcile.call_args.args[1])
            row = self.inbox.db.execute('SELECT * FROM messages WHERE id=9').fetchone()
            self.assertIn('收到测试', row['reply'])
            handle(self.inbox, row, config, API(), self.state, self.state, 'codex')
            self.assertEqual(transcribe.call_count, 1)

    def test_failed_reconciliation_remains_visible_in_status(self):
        class API:
            def send_message(self, *args):
                return {'message_id': 32}
        config = {'owner_chat_id': 123}
        self.inbox.receive([update(11, '昨天参加那个讲座了，python没开始呢')], config)
        report = {'disposition': 'reconciliation_unavailable', 'failure_code': 'vault_check_timeout', 'logged': False}
        with patch('reconciliation.process_reply', return_value=('学习库访问检查超时。', report)):
            handle(self.inbox, self.inbox.pending()[0], config, API(), self.state, self.state, 'codex')
        health = json.loads((self.state / 'reconciliation-health.json').read_text())
        self.assertEqual(health['failure_code'], 'vault_check_timeout')
        self.inbox.receive([update(12, '/status')], config)
        handle(self.inbox, self.inbox.pending()[0], config, API(), self.state, self.state, 'codex')
        reply = self.inbox.db.execute('SELECT reply FROM messages WHERE id=12').fetchone()[0]
        self.assertIn('最近一次对账未完成', reply)
        self.assertNotIn('进展会记入', reply)

    def test_unclear_voice_never_reaches_task_reconciliation(self):
        class API:
            def send_message(self, *args):
                return {'message_id': 31}
        config = {'owner_chat_id': 123}
        event = update(10)
        del event['message']['text']
        event['message']['voice'] = {'file_id': 'voice-id'}
        self.inbox.receive([event], config)
        with patch('bridge.transcribe_message', side_effect=VoiceError('没有听清')), \
             patch('reconciliation.process_reply') as reconcile:
            handle(self.inbox, self.inbox.pending()[0], config, API(), self.state, self.state, 'codex')
            reconcile.assert_not_called()
        self.assertEqual(self.inbox.db.execute('SELECT reply FROM messages WHERE id=10').fetchone()[0], '没有听清')


if __name__ == '__main__':
    unittest.main()
