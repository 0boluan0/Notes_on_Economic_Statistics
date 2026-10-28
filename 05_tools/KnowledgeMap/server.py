#!/usr/bin/env python3
"""Local, read-only vault projection plus a private course-link notebook."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import parse_qs, quote, unquote, urlsplit
from uuid import uuid4

HERE = Path(__file__).resolve().parent
PROGRESS_PATH = Path('99_学习情况记录/学习计划/MIT 18.06 Linear Algebra.md')
MAX_BODY = 32768
STATIC_TYPES = {'app.js': 'text/javascript; charset=utf-8', 'styles.css': 'text/css; charset=utf-8',
                'style.css': 'text/css; charset=utf-8'}


class APIError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def now():
    return datetime.now(timezone.utc).isoformat()


def scalar(value):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            return str(json.loads(value))
        except (ValueError, TypeError):
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return re.split(r'\s+#', value, maxsplit=1)[0].strip()


def flow_list(value):
    """Parse the vault's scalar YAML lists without executing a YAML loader."""
    inner = value.strip()[1:-1]
    items, start, quoted, escaped, depth = [], 0, None, False, 0
    for i, char in enumerate(inner):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\' and quoted == '"':
                escaped = True
            elif char == quoted:
                quoted = None
        elif char in "\"'":
            quoted = char
        elif char in '[{':
            depth += 1
        elif char in ']}':
            depth -= 1
        elif char == ',' and depth == 0:
            items.append(scalar(inner[start:i]))
            start = i + 1
    if inner[start:].strip():
        items.append(scalar(inner[start:]))
    return items


def frontmatter(text):
    """Read selected top-level metadata; leave all note content untouched."""
    match = re.match(r'\A\ufeff?---\s*\n(.*?)\n---\s*(?:\n|$)', text, re.S)
    if not match:
        return {}, text
    metadata, current = {}, None
    for line in match.group(1).splitlines():
        key = re.match(r'^([A-Za-z_][\w-]*):\s*(.*)$', line)
        if key:
            current, value = key.groups()
            if current not in {'atom_id', 'status', 'aliases', 'requires'}:
                current = None
                continue
            if current in {'aliases', 'requires'}:
                metadata[current] = flow_list(value) if value.strip().startswith('[') and value.strip().endswith(']') else ([scalar(value)] if value.strip() else [])
            else:
                metadata[current] = scalar(value)
        elif current in {'aliases', 'requires'}:
            item = re.match(r'^\s*-\s+(.+)$', line)
            if item:
                metadata[current].append(scalar(item.group(1)))
    return metadata, text[match.end():]


def clean_text(data, name, maximum=240, required=True):
    value = data.get(name, '')
    if not isinstance(value, str) or len(value) > maximum or '\x00' in value:
        raise APIError(400, f'Invalid {name}.')
    value = value.strip()
    if required and not value:
        raise APIError(400, f'{name} is required.')
    return value


def clean_url(data, name='url', required=True):
    value = clean_text(data, name, 2048, required)
    if not value and not required:
        return ''
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in {'https', 'http'} and parsed.hostname and
                 parsed.username is None and parsed.password is None)
        parsed.port
    except ValueError:
        valid = False
    if not valid or any(ord(c) < 33 or c in '\\<>"' for c in value):
        raise APIError(400, 'Use a complete http or https URL without credentials.')
    return value


def atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class KnowledgeMap:
    def __init__(self, vault_root, data_dir=None, seed_path=None, index_path=None):
        self.vault = Path(vault_root).resolve()
        self.notes_dir = self.vault / '00_Knowledge'
        self.data_dir = Path(data_dir).resolve() if data_dir else self.vault / '99_学习情况记录/knowledge-map'
        self.seed_path = Path(seed_path or HERE / 'seed.json').resolve()
        self.index_path = Path(index_path or HERE / 'static/index.html').resolve()
        self.lock = threading.RLock()
        self.note_cache = {}

    def seed(self):
        try:
            result = json.loads(self.seed_path.read_text(encoding='utf-8'))
            if not isinstance(result, dict):
                raise ValueError('Seed must be an object')
            for key in ('topics', 'relations', 'courses', 'mappings', 'domains'):
                if not isinstance(result.get(key), list):
                    raise ValueError('Missing seed list: ' + key)
            return result
        except (OSError, ValueError) as error:
            raise APIError(500, 'The map seed could not be read.') from error

    def state(self):
        path = self.data_dir / 'state.json'
        if not path.exists():
            return {'revision': 0, 'courses': [], 'mappings': [], 'note_links': []}
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(state, dict) or type(state.get('revision')) is not int or state['revision'] < 0:
                raise ValueError('Invalid state')
            if not all(isinstance(state.get(key), list) for key in ('courses', 'mappings', 'note_links')):
                raise ValueError('Invalid state lists')
            for key in ('courses', 'mappings', 'note_links'):
                if not all(isinstance(item, dict) and isinstance(item.get('id'), str) for item in state[key]):
                    raise ValueError('Invalid state records')
            return state
        except (OSError, ValueError) as error:
            raise APIError(500, 'Private state could not be read; it has been left unchanged.') from error

    @contextmanager
    def write_lock(self):
        with self.lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with (self.data_dir / '.state.lock').open('a') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def persist(self, state):
        path = self.data_dir / 'state.json'
        if path.exists():
            atomic_write(self.data_dir / 'state.previous.json', path.read_bytes())
        atomic_write(path, (json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

    def uri(self, relative):
        return 'obsidian://open?path=' + quote(str(self.vault / relative), safe='')

    def notes(self):
        """Only direct, nonsymlink Markdown files in 00_Knowledge are indexed."""
        with self.lock:
            result, live = {}, set()
            if self.notes_dir.is_symlink():
                return result
            for path in sorted(self.notes_dir.glob('*.md')):
                if path.is_symlink() or not path.is_file():
                    continue
                live.add(path)
                stat = path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
                cached = self.note_cache.get(path)
                if not cached or cached[0] != stamp:
                    text = path.read_text(encoding='utf-8')
                    metadata, body = frontmatter(text)
                    relative = path.relative_to(self.vault).as_posix()
                    atom_id = metadata.get('atom_id')
                    heading = re.search(r'^#\s+(.+)$', body, re.M)
                    note = {'id': atom_id or 'path:' + relative, 'id_kind': 'atom_id' if atom_id else 'path',
                            'title': path.stem, 'heading': heading.group(1) if heading else path.stem,
                            'aliases': metadata.get('aliases', []), 'requires': metadata.get('requires', []),
                            'status': metadata.get('status', 'unrecorded'), 'path': relative,
                            'uri': self.uri(relative), 'excerpt': body[:6000]}
                    self.note_cache[path] = (stamp, note)
                note = self.note_cache[path][1]
                if note['id'] in result:
                    raise APIError(409, 'Duplicate atom_id in 00_Knowledge; resolve it before linking notes.')
                result[note['id']] = note
            for path in self.note_cache.keys() - live:
                del self.note_cache[path]
            return result

    @staticmethod
    def metadata(note):
        return {key: value for key, value in note.items() if key != 'excerpt'}

    def search(self, query='', limit=30):
        terms = query.casefold().split()
        notes = [note for note in self.notes().values() if all(term in '\n'.join(
            [note['id'], note['title'], note['heading'], *note['aliases']]).casefold() for term in terms)]
        return {'notes': [self.metadata(note) for note in notes[:limit]], 'total': len(notes)}

    def progress(self):
        path = self.vault / PROGRESS_PATH
        result = {'progress_key': 'mit1806', 'course_id': 'mit-1806sc', 'source_path': PROGRESS_PATH.as_posix(),
                  'uri': self.uri(PROGRESS_PATH), 'source_label': 'MIT 18.06 学习计划 · 课程清单任务完成',
                  'read_at': now(), 'available': False, 'completed': None, 'total': None, 'mtime': None}
        if not path.is_file() or path.is_symlink():
            return result
        text = path.read_text(encoding='utf-8')
        section = re.search(r'^## 课程清单\s*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
        if section:
            checks = re.findall(r'^- \[([ xX])\] MIT 18\.06｜\d+\.\d+\s', section.group(1), re.M)
            result.update(available=True, completed=sum(c.lower() == 'x' for c in checks), total=len(checks),
                          mtime=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
        return result

    def map(self):
        seed, state, notes = self.seed(), self.state(), self.notes()
        result = dict(seed)
        for key in ('courses', 'mappings', 'note_links'):
            result[key] = seed.get(key, []) + state[key]
        linked_ids = {atom for topic in seed['topics'] for atom in topic.get('atom_ids', [])}
        linked_ids.update(link['atom_id'] for link in result['note_links'])
        result['notes_by_id'] = {nid: self.metadata(notes[nid]) for nid in sorted(linked_ids) if nid in notes}
        result['missing_note_ids'] = sorted(linked_ids - notes.keys())
        result['revision'] = state['revision']
        result['stats'] = {'topics': len(result['topics']), 'relations': len(result['relations']),
                           'courses': len(result['courses']), 'mappings': len(result['mappings']),
                           'notes': len(notes), 'linked_notes': len(result['notes_by_id'])}
        result['course_progress'] = self.progress()
        return result

    def mutate(self, resource, data, delete_id=None):
        allowed = {'courses': {'title', 'institution', 'version', 'url'},
                   'mappings': {'topic_id', 'course_id', 'title', 'url', 'role', 'statement'},
                   'note_links': {'topic_id', 'atom_id'}}
        if resource not in allowed or set(data) - allowed[resource] - {'revision'}:
            raise APIError(400, 'Unknown request fields.')
        with self.write_lock():
            state, seed = self.state(), self.seed()
            if 'revision' in data and (type(data['revision']) is not int or data['revision'] != state['revision']):
                raise APIError(409, 'The map changed. Refresh before saving again.')
            if delete_id:
                saved = next((x for x in state['mappings'] if x['id'] == delete_id), None)
                if saved is None:
                    raise APIError(404, 'Only a user-added mapping can be removed.')
                state['mappings'].remove(saved)
            elif resource == 'courses':
                saved = {'id': 'user-course-' + uuid4().hex, 'title': clean_text(data, 'title'),
                         'institution': clean_text(data, 'institution', 120, False),
                         'version': clean_text(data, 'version', 120), 'url': clean_url(data), 'source': 'user'}
                state['courses'].append(saved)
            else:
                topic_id = clean_text(data, 'topic_id')
                if topic_id not in {topic['id'] for topic in seed['topics']}:
                    raise APIError(400, 'Unknown topic_id.')
                if resource == 'mappings':
                    course_id = clean_text(data, 'course_id')
                    if course_id not in {course['id'] for course in seed['courses'] + state['courses']}:
                        raise APIError(400, 'Unknown course_id.')
                    role = clean_text(data, 'role')
                    if role not in {'teach', 'use', 'require'}:
                        raise APIError(400, 'role must be teach, use or require.')
                    saved = {'id': 'user-mapping-' + uuid4().hex, 'topic_id': topic_id, 'course_id': course_id,
                             'title': clean_text(data, 'title'), 'url': clean_url(data, required=False),
                             'role': role, 'statement': clean_text(data, 'statement', 2000), 'verification': 'user'}
                else:
                    atom_id = clean_text(data, 'atom_id', 1000)
                    if atom_id not in self.notes():
                        raise APIError(400, 'Unknown atom_id.')
                    existing = next((link for link in seed.get('note_links', []) + state['note_links']
                                     if link['topic_id'] == topic_id and link['atom_id'] == atom_id), None)
                    if existing:
                        return {'saved': existing, 'revision': state['revision']}
                    saved = {'id': 'user-note-link-' + uuid4().hex, 'topic_id': topic_id,
                             'atom_id': atom_id, 'verification': 'user'}
                state[resource].append(saved)
            state['revision'] += 1
            self.persist(state)
            return {'saved': saved, 'revision': state['revision']}


class Handler(BaseHTTPRequestHandler):
    server_version = 'KnowledgeMap/1'

    def log_message(self, format, *args):
        pass

    def send_bytes(self, status, content, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(content)

    def json(self, status, result):
        self.send_bytes(status, json.dumps(result, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def trusted_request(self, write=False):
        port = self.server.server_address[1]
        host = self.headers.get('Host', '')
        if host not in {f'127.0.0.1:{port}', f'localhost:{port}'}:
            raise APIError(403, 'Invalid local Host.')
        if write and self.headers.get('Origin') != 'http://' + host:
            raise APIError(403, 'A same-origin request is required.')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise APIError(403, 'Cross-site requests are not accepted.')

    def body(self):
        if self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding'):
            raise APIError(415, 'Use application/json with a Content-Length.')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            raise APIError(400, 'Invalid Content-Length.')
        if length <= 0 or length > MAX_BODY:
            raise APIError(413, 'Request body is empty or too large.')
        try:
            data = json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeError):
            raise APIError(400, 'Invalid JSON.')
        if not isinstance(data, dict):
            raise APIError(400, 'JSON body must be an object.')
        return data

    def dispatch(self, method):
        try:
            self.trusted_request(write=method != 'GET')
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if parsed.netloc or '\\' in path or '\x00' in path or '..' in path.split('/'):
                raise APIError(404, 'Not found.')
            query = parse_qs(parsed.query)
            app = self.server.app
            if method == 'GET':
                if path == '/api/map':
                    return self.json(200, app.map())
                if path == '/api/notes':
                    try:
                        limit = max(1, min(100, int(query.get('limit', ['30'])[0])))
                    except ValueError:
                        raise APIError(400, 'Invalid limit.')
                    return self.json(200, app.search(query.get('q', [''])[0][:500], limit))
                if path == '/api/note':
                    note = app.notes().get(query.get('id', [''])[0])
                    if note is None:
                        raise APIError(404, 'Note not found in 00_Knowledge.')
                    return self.json(200, note)
                if path in {'/', '/index.html'}:
                    file, mime = app.index_path, 'text/html; charset=utf-8'
                elif path[1:] in STATIC_TYPES:
                    file, mime = app.index_path.parent / path[1:], STATIC_TYPES[path[1:]]
                else:
                    raise APIError(404, 'Not found.')
                if file.is_symlink() or not file.is_file():
                    raise APIError(404, 'Static file not found.')
                return self.send_bytes(200, file.read_bytes(), mime)
            if method == 'POST' and path in {'/api/courses', '/api/mappings', '/api/note-links'}:
                return self.json(201, app.mutate(path.rsplit('/', 1)[1].replace('-', '_'), self.body()))
            if method == 'DELETE' and re.fullmatch(r'/api/mappings/[A-Za-z0-9_-]+', path):
                return self.json(200, app.mutate('mappings', {}, path.rsplit('/', 1)[1]))
            raise APIError(404, 'Not found.')
        except APIError as error:
            self.json(error.status, {'error': error.message})
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.json(500, {'error': 'Local data could not be read. Existing files were not replaced.'})

    def do_GET(self):
        self.dispatch('GET')

    def do_POST(self):
        self.dispatch('POST')

    def do_DELETE(self):
        self.dispatch('DELETE')


def create_server(vault_root, port=8768, data_dir=None, seed_path=None, index_path=None):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.app = KnowledgeMap(vault_root, data_dir, seed_path, index_path)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--vault-root', type=Path, default=HERE.parents[1])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--seed-path', type=Path)
    parser.add_argument('--index-path', type=Path)
    args = parser.parse_args()
    server = create_server(args.vault_root, args.port, args.data_dir, args.seed_path, args.index_path)
    print(f'Knowledge Map: http://127.0.0.1:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
