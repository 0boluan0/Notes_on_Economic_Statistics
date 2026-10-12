import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from app_server import Cancelled
from state import DEADLINES, RULES, rules, snapshot
from steward import Store, attach, completed_turn, prepare, register_claudian


REQUEST = {"scope": "synthetic", "title": "Acceptance classroom", "prompt": "Prepare one practice question and wait.", "model": "test-model"}


class FakeServer:
    def __init__(self):
        self.creates = self.starts = self.resumes = 0
        self.turns = []
        self.crash = None
        self.on_finish = lambda: None

    def __call__(self, cancelled):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def create(self, model):
        self.creates += 1
        if self.crash == "after_create":
            raise SystemExit("process interrupted after native creation")
        return "native-thread"

    def resume(self, thread_id):
        assert thread_id == "native-thread"
        self.resumes += 1

    def start(self, thread_id, prompt):
        self.starts += 1
        if self.crash == "before_turn_reply":
            raise SystemExit("turn send outcome unknown")
        return "native-turn"

    def read(self, thread_id):
        return {"id": thread_id, "turns": self.turns}

    def finish(self, thread_id, turn_id):
        self.turns = [{"id": turn_id, "status": "completed", "items": [{"type": "agentMessage", "text": "Ready for your first answer."}]}]
        self.on_finish()
        if self.crash == "after_turn":
            raise SystemExit("process interrupted before local result receipt")
        return self.read(thread_id)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.open_store()
        self.server = FakeServer()
        self.store.enqueue("lesson-1", REQUEST)

    def open_store(self):
        self.store = Store(self.root / "runtime", self.vault, self.root / "codex")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def restart_store(self):
        self.store.close()
        self.open_store()

    def test_same_operation_is_durable_and_different_input_is_rejected(self):
        first = prepare(self.store, "lesson-1", self.server)
        self.restart_store()
        self.store.enqueue("lesson-1", REQUEST)
        second = prepare(self.store, "lesson-1", self.server)
        self.assertEqual(first["thread_id"], second["thread_id"])
        self.assertEqual((self.server.creates, self.server.starts), (1, 1))
        with self.assertRaises(ValueError):
            self.store.enqueue("lesson-1", dict(REQUEST, prompt="different"))

    def test_crash_after_turn_recovers_result_without_new_turn(self):
        self.server.crash = "after_turn"
        with self.assertRaises(SystemExit):
            prepare(self.store, "lesson-1", self.server)
        self.restart_store()
        self.server.crash = None
        job = prepare(self.store, "lesson-1", self.server)
        self.assertEqual(job["state"], "prepared")
        self.assertEqual((self.server.creates, self.server.starts, self.server.resumes), (1, 1, 1))
        self.assertIn("first answer", job["result"])

    def test_unknown_creation_stops_instead_of_creating_duplicate(self):
        self.server.crash = "after_create"
        with self.assertRaises(SystemExit):
            prepare(self.store, "lesson-1", self.server)
        self.restart_store()
        job = prepare(self.store, "lesson-1", self.server)
        self.assertEqual(job["state"], "uncertain")
        self.assertEqual(self.server.creates, 1)

    def test_unknown_turn_send_does_not_blindly_resubmit(self):
        self.server.crash = "before_turn_reply"
        with self.assertRaises(SystemExit):
            prepare(self.store, "lesson-1", self.server)
        self.restart_store()
        self.server.crash = None
        job = prepare(self.store, "lesson-1", self.server)
        self.assertEqual(job["state"], "uncertain")
        self.assertEqual((self.server.creates, self.server.starts), (1, 1))

    def test_cancel_discards_late_completion_and_keeps_audit(self):
        self.server.on_finish = lambda: self.store.cancel("lesson-1")
        job = prepare(self.store, "lesson-1", self.server)
        self.assertEqual(job["state"], "cancelled")
        self.assertIsNone(job["result"])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM events WHERE state='late_result_ignored'").fetchone()[0], 1)
        self.assertEqual(prepare(self.store, "lesson-1", self.server)["state"], "cancelled")
        self.assertEqual(self.server.creates, 1)

    def test_late_native_id_after_cancel_is_preserved_in_audit(self):
        self.store.cancel("lesson-1")
        with self.assertRaises(Cancelled):
            self.store.update("lesson-1", "preparing", thread_id="late-native-id")
        record = self.store.db.execute("SELECT detail FROM events WHERE state='late_result_ignored'").fetchone()[0]
        self.assertEqual(json.loads(record)["thread_id"], "late-native-id")

    def test_registration_can_resume_after_lost_receipt(self):
        prepare(self.store, "lesson-1", self.server)
        registered = set()
        calls = []

        def register(vault, thread_id, title, model):
            registered.add(thread_id)
            calls.append(thread_id)
            if len(calls) == 1:
                raise TimeoutError("UI registered but response was lost")
            return {"conversation_id": thread_id}

        self.assertEqual(attach(self.store, "lesson-1", self.vault, register)["state"], "linking")
        self.restart_store()
        job = attach(self.store, "lesson-1", self.vault, register)
        self.assertEqual(job["state"], "registered")
        self.assertEqual(registered, {"native-thread"})
        attach(self.store, "lesson-1", self.vault, register)
        self.assertEqual(len(calls), 2)

    def test_cancelled_request_cannot_be_attached(self):
        prepare(self.store, "lesson-1", self.server)
        self.store.cancel("lesson-1")
        def forbidden(*args):
            self.fail("cancelled request was registered")
        self.assertEqual(attach(self.store, "lesson-1", self.vault, forbidden)["state"], "cancelled")

    def test_real_progress_scope_and_vault_runtime_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.enqueue("real", dict(REQUEST, scope="real"))
        with self.assertRaises(ValueError):
            Store(self.vault / "runtime", self.vault, self.root / "codex")

    def test_mismatched_or_tool_using_history_is_not_prepared(self):
        for turn in [
            {"id": "other", "status": "completed", "items": []},
            {"id": "expected", "status": "inProgress", "items": []},
            {"id": "expected", "status": "completed", "items": [{"type": "commandExecution"}]},
        ]:
            with self.assertRaises(RuntimeError):
                completed_turn({"turns": [turn]}, "expected")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name)
        self.write(rules.OVERVIEW, "# Overall\n## Active\n- 状态：active\n- 真实计划：[[active]]\n## Paused\n- 状态：paused\n- 真实计划：[[paused]]\n")
        self.write(RULES, "# Student OS rules\n")
        self.write(rules.LOG, "# Prior reports\n")
        self.write("active.md", "- [ ] Unrecorded task #student-os/task ^one\n```md\n- [ ] Example #student-os/task\n```\n")
        self.write("paused.md", "- [ ] Paused task #student-os/task ^paused\n")
        self.write(DEADLINES, "- [ ] Past obligation #student-os/deadline 📅 2020-01-01 ^deadline\n- [ ] Archived #student-os/archived 📅 2020-01-01\n")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, content):
        p = self.vault / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def test_read_only_inventory_keeps_unknown_state_and_date_precision(self):
        before = {p: p.read_bytes() for p in self.vault.rglob("*.md")}
        result = snapshot(self.vault)
        self.assertEqual(result["status"], "read")
        self.assertEqual(len(result["tasks"]), 1)
        self.assertEqual(result["tasks"][0]["ref"], "active.md#^one")
        self.assertEqual(result["tasks"][0]["status"], "completion_unknown")
        self.assertEqual(len(result["obligations"]), 1)
        self.assertIsNone(result["obligations"][0]["local_time"])
        self.assertLess(result["obligations"][0]["days_until"], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.vault.rglob("*.md")})

    def test_missing_deadlines_is_a_visible_gap(self):
        (self.vault / DEADLINES).unlink()
        result = snapshot(self.vault)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(any(i.get("path") == DEADLINES for i in result["issues"]))

    def test_missing_registered_source_is_not_silent(self):
        (self.vault / "active.md").unlink()
        self.assertTrue(any(i["code"] == "unresolved_registered_source" for i in snapshot(self.vault)["issues"]))

    def test_duplicate_identity_is_not_usable_for_dispatch(self):
        self.write("active.md", "- [ ] One #student-os/task ^same\n- [ ] Two #student-os/task ^same\n")
        result = snapshot(self.vault)
        self.assertTrue(all(t["ref"] is None for t in result["tasks"]))
        self.assertTrue(any(i["code"] == "duplicate_task_identity" for i in result["issues"]))


class AdapterTests(unittest.TestCase):
    def test_empty_write_reply_requires_independent_native_history_readback(self):
        reply = {"conversation_id": "native", "thread_id": "native", "title": "Class", "message_count": 2}
        with patch("steward.subprocess.run", side_effect=[
            SimpleNamespace(returncode=0, stdout=""),
            SimpleNamespace(returncode=0, stdout="=> " + json.dumps(reply)),
        ]) as run:
            self.assertEqual(register_claudian(Path("/tmp"), "native", "Class", "test-model"), reply)
        self.assertEqual(run.call_count, 2)

    def test_mismatched_readback_does_not_report_success(self):
        reply = {"thread_id": "different", "title": "Class", "message_count": 2}
        with patch("steward.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="=> " + json.dumps(reply))), patch("steward.time.sleep"):
            with self.assertRaises(RuntimeError):
                register_claudian(Path("/tmp"), "native", "Class", "test-model")


if __name__ == "__main__":
    unittest.main()
