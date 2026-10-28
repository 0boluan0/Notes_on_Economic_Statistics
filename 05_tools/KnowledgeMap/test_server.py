"""Boundary tests use temporary vaults; never write the real private state."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import quote

from server import APIError, HERE, KnowledgeMap, MAX_BODY, PROGRESS_PATH, create_server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        notes = self.root / '00_Knowledge'
        notes.mkdir()
        (notes / '向量.md').write_text('---\natom_id: LA-001\nstatus: source-checked\naliases:\n  - "Vector, linear"\n  - 向量别名\nrequires:\n  - "[[加法]]"\n---\n# A vector\n<img src=x onerror=alert(1)>\n' + 'x' * 7000, encoding='utf-8')
        (notes / '旧笔记.md').write_text('---\naliases: ["Legacy, note", old]\nrequires: ["[[集合]]", "[[映射|mapping]]"]\n---\n# Legacy note\nText', encoding='utf-8')
        (notes / '_hubs').mkdir()
        (notes / '_hubs' / 'nested.md').write_text('Do not index nested notes')
        (self.root / '08_quant').mkdir()
        secret = self.root / '08_quant' / 'secret.md'
        secret.write_text('PRIVATE STRATEGY')
        (notes / 'linked-secret.md').symlink_to(secret)
        plan = self.root / PROGRESS_PATH
        plan.parent.mkdir(parents=True)
        plan.write_text('# Test\n- [x] Unrelated task\n## 课程清单\n- [x] MIT 18.06｜1.1 One\n- [ ] MIT 18.06｜1.2 Two\n## Other\n- [x] MIT 18.06｜2.1 Outside\n', encoding='utf-8')
        self.plan_before = plan.read_bytes()
        self.seed = self.root / 'seed.json'
        self.seed.write_text(json.dumps({'topics': [{'id': 'vector', 'atom_ids': ['LA-001']}],
            'relations': [], 'courses': [{'id': 'mit-1806sc', 'title': 'Linear Algebra', 'version': '2011'}],
            'mappings': [{'id': 'seed-mapping', 'topic_id': 'vector', 'course_id': 'mit-1806sc'}],
            'domains': []}), encoding='utf-8')
        self.data = self.root / 'private-state'
        static = self.root / 'static'
        static.mkdir()
        self.index = static / 'index.html'
        self.index.write_text('<!doctype html><p>Map</p>')
        (static / 'app.js').write_text('console.log("map");')
        self.app = KnowledgeMap(self.root, self.data, self.seed, self.index)

    def course(self, **changes):
        return {'title': 'A course', 'institution': 'Example', 'version': '2026',
                'url': 'https://example.edu/course', **changes}

    def mapping(self, **changes):
        return {'topic_id': 'vector', 'course_id': 'mit-1806sc', 'title': 'Chapter 1',
                'url': 'https://example.edu/chapter', 'role': 'teach',
                'statement': 'Defines vectors.', **changes}

    def test_search_is_bounded_and_progress_is_live(self):
        result = self.app.search('linear')
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['notes'][0]['requires'], ['[[加法]]'])
        self.assertNotIn('excerpt', result['notes'][0])
        notes = self.app.notes()
        self.assertEqual(len(notes), 2)
        fallback = notes['path:00_Knowledge/旧笔记.md']
        self.assertEqual(fallback['id_kind'], 'path')
        self.assertEqual(fallback['aliases'], ['Legacy, note', 'old'])
        self.assertEqual(fallback['requires'], ['[[集合]]', '[[映射|mapping]]'])
        self.assertEqual(len(notes['LA-001']['excerpt']), 6000)
        self.assertNotIn('atom_id:', notes['LA-001']['excerpt'])
        self.assertIn('obsidian://open?path=', notes['LA-001']['uri'])
        initial = self.app.map()
        self.assertEqual(initial['course_progress']['completed'], 1)
        self.assertEqual(initial['course_progress']['total'], 2)
        self.assertEqual(set(initial['notes_by_id']), {'LA-001'})
        self.assertFalse(self.data.exists(), 'GET must not create private state')
        plan = self.root / PROGRESS_PATH
        plan.write_text(plan.read_text().replace('[ ]', '[x]'))
        self.assertEqual(self.app.progress()['completed'], 2)
        self.assertFalse(self.data.exists())

    def test_url_validation_text_data_and_unknown_references(self):
        literal = '<img src=x onerror="alert(1)">'
        saved = self.app.mutate('courses', self.course(title=literal))['saved']
        self.assertEqual(saved['title'], literal)
        for bad in ['javascript:alert(1)', 'data:text/html,x', 'https://user:pass@example.edu',
                    'https://example.edu\\@evil.test', 'https://example.edu/\nscript', '/local/path']:
            with self.subTest(url=bad), self.assertRaises(APIError):
                self.app.mutate('courses', self.course(url=bad))
        for changes in [{'course_id': 'unknown'}, {'topic_id': 'unknown'}, {'role': 'mastered'}]:
            with self.subTest(changes=changes), self.assertRaises(APIError):
                self.app.mutate('mappings', self.mapping(**changes))
        with self.assertRaises(APIError):
            self.app.mutate('note_links', {'topic_id': 'vector', 'atom_id': 'path:08_quant/secret.md'})
        self.assertEqual(len(self.app.state()['courses']), 1)
        self.assertEqual(self.app.state()['mappings'], [])

    def test_persistence_restart_backup_and_seed_delete_boundary(self):
        created = self.app.mutate('mappings', self.mapping())
        state_path = self.data / 'state.json'
        first_bytes = state_path.read_bytes()
        self.app.mutate('note_links', {'topic_id': 'vector', 'atom_id': 'path:00_Knowledge/旧笔记.md', 'revision': 1})
        self.assertEqual((self.data / 'state.previous.json').read_bytes(), first_bytes)
        restarted = KnowledgeMap(self.root, self.data, self.seed, self.index)
        self.assertEqual(restarted.state()['revision'], 2)
        self.assertIn('path:00_Knowledge/旧笔记.md', restarted.map()['notes_by_id'])
        with self.assertRaises(APIError) as conflict:
            restarted.mutate('courses', self.course(revision=0))
        self.assertEqual(conflict.exception.status, 409)
        with self.assertRaises(APIError):
            restarted.mutate('mappings', {}, 'seed-mapping')
        restarted.mutate('mappings', {}, created['saved']['id'])
        self.assertEqual(restarted.state()['mappings'], [])
        self.assertEqual(len(restarted.map()['mappings']), 1)
        self.assertEqual((self.root / PROGRESS_PATH).read_bytes(), self.plan_before)
        def simultaneous_write(number):
            independent = KnowledgeMap(self.root, self.data, self.seed, self.index)
            return independent.mutate('courses', self.course(title=f'Course {number}'))
        with ThreadPoolExecutor(max_workers=6) as workers:
            list(workers.map(simultaneous_write, range(6)))
        self.assertEqual(restarted.state()['revision'], 9)
        self.assertEqual(len(restarted.state()['courses']), 6, 'Concurrent writers must not lose records')
        state_path.write_text('{broken')
        with self.assertRaises(APIError):
            restarted.mutate('courses', self.course())
        self.assertEqual(state_path.read_text(), '{broken', 'Corrupt state must not be reset')

    def test_http_rejects_traversal_cross_origin_and_large_body(self):
        server = create_server(self.root, 0, self.data, self.seed, self.index)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        origin = f'http://127.0.0.1:{port}'

        def request(method, path, body=None, headers=None):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            try:
                connection.request(method, path, body, headers or {})
                response = connection.getresponse()
                return response.status, response.getheader('Content-Type'), response.read()
            finally:
                connection.close()

        for path in ['/../seed.json', '/%2e%2e/seed.json', '/static/../seed.json', '/seed.json', '/server.py',
                     '/api/note?id=' + quote('path:08_quant/secret.md')]:
            with self.subTest(path=path):
                self.assertEqual(request('GET', path)[0], 404)
        self.assertEqual(request('GET', '/', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(request('GET', '/')[0], 200)
        body = json.dumps(self.course(title='<script>alert(1)</script>'))
        good_headers = {'Origin': origin, 'Content-Type': 'application/json'}
        self.assertEqual(request('POST', '/api/courses', body, {'Content-Type': 'application/json'})[0], 403)
        self.assertEqual(request('POST', '/api/courses', body, {**good_headers, 'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(request('POST', '/api/courses', body, {'Origin': origin, 'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(request('POST', '/api/courses', 'x' * (MAX_BODY + 1), good_headers)[0], 413)
        status, content_type, response = request('POST', '/api/courses', body, good_headers)
        self.assertEqual(status, 201)
        self.assertTrue(content_type.startswith('application/json'))
        self.assertEqual(json.loads(response)['saved']['title'], '<script>alert(1)</script>')

    def test_real_seed_references_and_prerequisite_graph(self):
        real_seed = HERE / 'seed.json'
        if not real_seed.exists():
            self.skipTest('The map seed has not been authored yet')
        seed = json.loads(real_seed.read_text())
        topics = {x['id'] for x in seed['topics']}
        courses = {x['id'] for x in seed['courses']}
        self.assertEqual(len(topics), len(seed['topics']))
        self.assertEqual(len(courses), len(seed['courses']))
        graph = {tid: [] for tid in topics}
        for edge in seed['relations']:
            self.assertIn(edge['from'], topics)
            self.assertIn(edge['to'], topics)
            self.assertTrue(edge['evidence']['statement'])
            if edge['kind'] == 'prerequisite':
                graph[edge['from']].append(edge['to'])
        visited, visiting = set(), set()

        def visit(node):
            self.assertNotIn(node, visiting, f'Prerequisite cycle at {node}')
            if node in visited:
                return
            visiting.add(node)
            for target in graph[node]:
                visit(target)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)
        for mapping in seed['mappings']:
            self.assertIn(mapping['topic_id'], topics)
            self.assertIn(mapping['course_id'], courses)
            self.assertIn(mapping['role'], {'teach', 'use', 'require'})
        actual_vault = HERE.parents[1]
        if (actual_vault / '00_Knowledge').is_dir():
            actual_notes = KnowledgeMap(actual_vault, self.data, real_seed, self.index).notes()
            for topic in seed['topics']:
                for atom_id in topic.get('atom_ids', []):
                    self.assertIn(atom_id, actual_notes, f'Unknown seed atom in {topic["id"]}')


if __name__ == '__main__':
    unittest.main()
