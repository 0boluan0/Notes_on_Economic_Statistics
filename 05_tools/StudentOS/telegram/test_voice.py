import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from voice import VoiceError, transcribe_message


class VoiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)
        (self.state / 'models').mkdir()
        (self.state / 'models/small.pt').touch()
        executable = self.state / 'whisper'
        executable.touch()
        patched = patch('voice.WHISPER', executable)
        patched.start()
        self.addCleanup(patched.stop)
        self.api = Mock()
        self.api.download_file.return_value = b'private audio'
        self.message = {'voice': {'file_id': 'voice-id', 'duration': 12, 'file_size': 50000}}

    def output(self, command, **kwargs):
        work = Path(command[command.index('--output_dir') + 1])
        self.assertEqual((work / 'input.audio').stat().st_mode & 0o777, 0o600)
        (work / 'input.json').write_text(json.dumps({
            'text': '收到测试', 'language': 'zh',
            'segments': [{'avg_logprob': -0.3, 'no_speech_prob': 0.01, 'compression_ratio': 1.2}],
        }))

    def test_success_caches_text_and_removes_raw_audio(self):
        with patch('voice.subprocess.run', side_effect=self.output) as run:
            self.assertEqual(transcribe_message(self.api, self.message, self.state, 9), '收到测试')
            self.assertEqual(transcribe_message(self.api, self.message, self.state, 9), '收到测试')
            self.assertEqual(run.call_count, 1)
        self.api.download_file.assert_called_once()
        self.assertFalse(list(self.state.glob('voice-*')))
        self.assertEqual(next((self.state / 'transcripts').glob('*.json')).stat().st_mode & 0o777, 0o600)

    def test_oversized_message_does_not_download(self):
        self.message['voice']['duration'] = 601
        with self.assertRaises(VoiceError):
            transcribe_message(self.api, self.message, self.state, 9)
        self.api.download_file.assert_not_called()

    def test_timeout_cleans_audio_and_contains_no_private_error(self):
        with patch('voice.subprocess.run', side_effect=subprocess.TimeoutExpired('secret', 300)):
            with self.assertRaises(VoiceError) as error:
                transcribe_message(self.api, self.message, self.state, 9)
        self.assertNotIn('secret', str(error.exception))
        self.assertFalse(list(self.state.glob('voice-*')))

    def test_low_confidence_is_not_cached_as_user_statement(self):
        def unclear(command, **kwargs):
            self.output(command, **kwargs)
            work = Path(command[command.index('--output_dir') + 1])
            result = json.loads((work / 'input.json').read_text())
            result['segments'][0]['avg_logprob'] = -1.5
            (work / 'input.json').write_text(json.dumps(result))
        with patch('voice.subprocess.run', side_effect=unclear):
            with self.assertRaises(VoiceError):
                transcribe_message(self.api, self.message, self.state, 9)
        self.assertFalse(list((self.state / 'transcripts').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
