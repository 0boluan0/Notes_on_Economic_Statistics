"""Transcribe owner-authorized audio locally; keep only a private text receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from telegram_api import TelegramError


WHISPER = Path('/opt/homebrew/bin/whisper')
MAX_BYTES = 20_000_000


class VoiceError(Exception):
    """A user-facing failure that contains neither credentials nor tool output."""


def transcribe_message(api, message, state, update_id):
    audio = message.get('voice') or message.get('audio') or {}
    if not audio.get('file_id'):
        raise VoiceError('这条语音没有可读取的音频，请重新发一次。')
    if audio.get('duration', 0) > 600 or audio.get('file_size', 0) > MAX_BYTES:
        raise VoiceError('请把语音分成每段不超过 10 分钟、20 MB 的几条发送。')
    state = Path(state)
    receipts = state / 'transcripts'
    receipts.mkdir(exist_ok=True, mode=0o700)
    identity = hashlib.sha256(f"{update_id}:{audio['file_id']}".encode()).hexdigest()
    cached = receipts / (identity + '.json')
    if cached.exists():
        return json.loads(cached.read_text())['text']
    if not WHISPER.is_file() or not (state / 'models/small.pt').is_file():
        raise VoiceError('本机语音识别暂时不可用，这条还没有更新任务。请稍后重发，或先发文字。')
    try:
        data = api.download_file(audio['file_id'], max_bytes=MAX_BYTES)
    except TelegramError:
        raise VoiceError('这条语音暂时下载失败，还没有更新任务。请稍后重发一次。') from None
    # The file name and command arguments contain no Telegram identifiers/token.
    with tempfile.TemporaryDirectory(prefix='voice-', dir=state) as temporary:
        work = Path(temporary)
        source = work / 'input.audio'
        source.write_bytes(data)
        source.chmod(0o600)
        try:
            subprocess.run([
                str(WHISPER), str(source), '--model', 'small',
                '--model_dir', str(state / 'models'), '--device', 'cpu',
                '--fp16', 'False', '--threads', '4', '--task', 'transcribe',
                '--output_format', 'json', '--output_dir', str(work),
                '--verbose', 'False', '--condition_on_previous_text', 'False',
            ], check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = json.loads((work / 'input.json').read_text())
        except (OSError, subprocess.SubprocessError, ValueError):
            raise VoiceError('这条语音转写没有成功，还没有更新任务。请稍后重发，或先发文字。') from None
    text = result.get('text', '').strip()
    segments = result.get('segments', [])
    if (not text or len(text) > 6000 or not segments
            or any(s.get('avg_logprob', -2) < -1.0
                   or s.get('no_speech_prob', 1) > 0.6
                   or s.get('compression_ratio', 3) > 2.4 for s in segments)):
        raise VoiceError('这条语音我没有听清，还没有更新任务。请再说一遍，或补一句文字。')
    temporary = cached.with_suffix('.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, 0o600)
        json.dump({'text': text, 'language': result.get('language'), 'model': 'whisper-small'}, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, cached)
    return text
